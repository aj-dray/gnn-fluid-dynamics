import collections, random, torch
from torch.utils.data import Sampler, BatchSampler, RandomSampler, SequentialSampler


class RolloutSampler(Sampler):
    """Simple sampler that orders samples by timestep for rollout"""

    def __init__(self, dataset, shuffle=False, num_rollout_trajectories=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_rollout_trajectories = num_rollout_trajectories
        self.indices = self._create_rollout_indices()

    def _create_rollout_indices(self):
        """Create indices grouped by timestep"""
        # Group samples by timestep
        timestep_groups = {}
        trajectory_set = set()

        for idx, (traj_id, timestep) in enumerate(self.dataset.sample_map):
            if timestep not in timestep_groups:
                timestep_groups[timestep] = []
            timestep_groups[timestep].append(idx)
            trajectory_set.add(traj_id)

        # Limit trajectories if specified
        if self.num_rollout_trajectories:
            trajectories = sorted(list(trajectory_set))[:self.num_rollout_trajectories]
            trajectory_set = set(trajectories)

        # Create rollout-ordered indices: all trajectories at timestep 0, then timestep 1, etc.
        rollout_indices = []
        for timestep in sorted(timestep_groups.keys()):
            timestep_indices = [idx for idx in timestep_groups[timestep]
                              if self.dataset.sample_map[idx][0] in trajectory_set]
            if self.shuffle:
                self.dataset.rng.shuffle(timestep_indices)
            rollout_indices.extend(timestep_indices)

        return rollout_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class MultiMeshBatchSampler(BatchSampler):

    def __init__(self, base_sampler, dataset,
                 batch_size: int,
                 num_meshes: int,
                 drop_last: bool = True):

        if batch_size % num_meshes:
            raise ValueError("`batch_size` must be divisible by `num_meshes` "
                             f"(got {batch_size=} and {num_meshes=}).")

        self.sampler        = base_sampler
        self.dataset        = dataset
        self.batch_size     = batch_size
        self.num_meshes     = num_meshes
        self.k_per_mesh     = batch_size // num_meshes
        self.drop_last      = drop_last

    def __iter__(self):
        buckets  = collections.defaultdict(list)   # mesh_id -> [idx…]
        cur_batch: list[int] = []

        for idx in self.sampler:
            mesh_id, _ = self.dataset.sample_map[idx]
            bucket = buckets[mesh_id]
            bucket.append(idx)

            # once we have k_per_mesh indices for this mesh -> move to batch
            if len(bucket) == self.k_per_mesh:
                cur_batch.extend(bucket)
                bucket.clear()

            # when batch is full (num_meshes × k) -> yield
            if len(cur_batch) == self.batch_size:
                yield cur_batch
                cur_batch = []

        if not self.drop_last and cur_batch:
            yield cur_batch

    def __len__(self) -> int:
        return len(self.sampler) // self.batch_size

class ChunkedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """
    Build batches of *exactly* ``batch_size`` samples that come from
    ``num_meshes`` different meshes.
    Each selected mesh contributes
        k = batch_size // num_meshes
    samples to every batch and stays in the active set for ``reuse``
    consecutive batches, so its geometry remains hot in the worker-side cache.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Must expose ``dataset.sample_map[idx] -> (mesh_id, timestep)``.
    batch_size : int
        Total number of samples in one physical batch.
    num_meshes : int
        How many different meshes must appear in every batch.
    reuse : int, default 1
        How many successive batches a mesh remains active
        (>=1 → 1 means simple round-robin).
    drop_last : bool, default True
        If True, discard a trailing incomplete batch at epoch end.
    generator : torch.Generator | None, default None
        Optional pyro RNG to control shuffling deterministically.
    """
    def __init__(self,
                 dataset,
                 batch_size: int,
                 num_meshes: int,
                 reuse: int = 1,
                 drop_last: bool = True,
                 generator: torch.Generator | None = None):

        if batch_size % num_meshes:
            raise ValueError("`batch_size` must be divisible by `num_meshes` "
                             f"(got batch_size={batch_size}, num_meshes={num_meshes}).")

        self.dataset    = dataset
        self.batch_size = batch_size
        self.num_meshes = num_meshes
        self.k          = batch_size // num_meshes
        self.reuse      = max(1, reuse)
        self.drop_last  = drop_last
        self.gen        = generator

        # build lookup: mesh_id -> tensor(indices)
        mesh2idx = collections.defaultdict(list)
        for idx, (mesh_id, _) in enumerate(dataset.sample_map):
            mesh2idx[mesh_id].append(idx)
        self.mesh_to_indices = {m: torch.tensor(v) for m, v in mesh2idx.items()}
        self.all_mesh_ids    = list(self.mesh_to_indices)

    # ------------------------------------------------------------------ #
    def __iter__(self):
        g = self.gen if self.gen is not None else torch.Generator()
        # shuffle meshes once per epoch
        mesh_queue = torch.randperm(len(self.all_mesh_ids), generator=g).tolist()

        active:  list[str]   = []   # currently cached meshes
        cycles:  dict[str,int] = {} # how many batches they've stayed
        while mesh_queue or active:
            # top-up active set up to num_meshes
            while len(active) < self.num_meshes and mesh_queue:
                m = self.all_mesh_ids[mesh_queue.pop()]
                active.append(m)
                cycles[m] = 0

            batch: list[int] = []
            for m in list(active):                 # iterate over a *copy*
                idx_pool = self.mesh_to_indices[m]
                chosen   = idx_pool[torch.randint(len(idx_pool),
                                                 (self.k,),
                                                 generator=g)]
                batch.extend(chosen.tolist())

                cycles[m] += 1
                if cycles[m] == self.reuse:        # time to evict
                    active.remove(m)
                    del cycles[m]

            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and batch:
                yield batch

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.dataset) // self.batch_size



class PerMeshBatchSampler(BatchSampler):
    """
    Groups indices produced by `base_sampler` so that
    *each yielded batch is drawn from a single mesh*.
    """
    def __init__(self, base_sampler, dataset, batch_size, drop_last=True):
        super().__init__(base_sampler, batch_size, drop_last)
        self.dataset     = dataset          # needs `sample_map`
        self.batch_size  = batch_size
        self.drop_last   = drop_last

    def __iter__(self):
        # buckets accumulate indices per mesh until they reach batch_size
        buckets = collections.defaultdict(list)

        for idx in self.sampler:            # <-- 'sampler' inherited from BatchSampler
            mesh_id, _ = self.dataset.sample_map[idx]
            bucket     = buckets[mesh_id]
            bucket.append(idx)

            if len(bucket) == self.batch_size:
                yield bucket[:]             # full batch
                bucket.clear()              # reset for that mesh

        # end-of-epoch: optionally yield leftovers
        if not self.drop_last:
            for bucket in buckets.values():
                if bucket:
                    yield bucket

    def __len__(self):
        # Count samples per mesh
        mesh_counts = collections.defaultdict(int)
        for mesh_id, _ in self.dataset.sample_map:
            mesh_counts[mesh_id] += 1

        # Calculate batches per mesh and sum
        total_batches = 0
        for count in mesh_counts.values():
            # Full batches from this mesh
            full_batches = count // self.batch_size
            total_batches += full_batches

            # Add partial batch if needed
            if not self.drop_last and count % self.batch_size > 0:
                total_batches += 1

        return total_batches


def get_sampler(dataset, parameters, random=True, optimise_order=False, drop_last=False):
    B = parameters.training.batch_size

    if optimise_order:
        # keep randomness but batch per mesh
        base = RandomSampler(dataset, replacement=False) if random \
               else SequentialSampler(dataset)
        # return PerMeshBatchSampler(base, dataset, B, drop_last)
        # return MultiMeshBatchSampler(base, dataset, B, 4, drop_last)
        # return ChunkedBatchSampler(dataset, B, 4, 4, drop_last=drop_last)

    # vanilla behaviour
    base = RandomSampler(dataset, replacement=False) if random \
           else SequentialSampler(dataset)
    return torch.utils.data.BatchSampler(base, B, drop_last)
