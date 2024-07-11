import logging
from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers.cache_utils import Cache, StaticCache
from transformers.configuration_utils import PretrainedConfig


class RobustCacheManagement(StaticCache):
    def __init__(self, config: PretrainedConfig, max_batch_size: int = 16, max_cache_len: int = 4096, device: torch.device = torch.device('cpu'), dtype=torch.float32) -> None:
        super().__init__(config, max_batch_size, max_cache_len, device, dtype)

    @dataclass(repr=False, eq=False)
    class CacheStatus:
        non_empty: List[int]
        empty: List[int]

    """
    An extension of StaticCache with added functionality for cache status checks and trimming.
    """

    def is_empty(self, layer_idx: int = 0) -> bool:
        """
        Checks if the cache at the given layer index is empty (all elements are zero).
        """
        # Check if all elements in the first head of the first sample are zero
        return not self.key_cache[layer_idx][0, 0].any() and not self.value_cache[layer_idx][0, 0].any()

    def get_non_empty_indices(self) -> CacheStatus:
        """
        Returns a CacheStatus object with two fields: non_empty and empty, 
        listing indices of non-empty and empty caches respectively.
        """
        non_empty = [layer_idx for layer_idx in range(
            len(self.key_cache)) if not self.is_empty(layer_idx)]
        empty = [layer_idx for layer_idx in range(
            len(self.key_cache)) if self.is_empty(layer_idx)]
        return self.CacheStatus(non_empty=non_empty, empty=empty)

    def trim_cache(self, start_index: Optional[int] = None, end_index: Optional[int] = None):
        """
        !TODO: Trims the cache starting from `start_index` up to `end_index`. If not specified, trims the entire cache.
        """
        pass

    def reset_sample_cache(self, sample_index: int):
        """
        Resets the cache for a specific sample index across all layers.
        Logs a warning if the sample index is out of bounds.
        """
        num_samples = self.key_cache[0].shape[0]
        if not 0 <= sample_index < num_samples:
            logging.warning(
                f"Sample index {sample_index} is out of bounds. Available samples are within [0, {num_samples}).")
            return

        for layer_idx in range(len(self)):
            # key and value like [batch_size, num_heads, seq_len, head_dim]
            self.key_cache[layer_idx][sample_index, :, :, :] = 0
            self.value_cache[layer_idx][sample_index, :, :, :] = 0


if __name__ == '__main__':
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create a configuration for the cache
    config = PretrainedConfig()
    config.num_hidden_layers = 3
    config.num_key_value_heads = 8
    config.hidden_size = 768
    config.num_attention_heads = 12
    config.max_position_embeddings = 4096
    max_batch_size = 2

    # Create an instance of RobustCacheManagement
    cache_manager = RobustCacheManagement(
        config, max_batch_size=max_batch_size, max_cache_len=2048, device=torch.device('cpu'), dtype=torch.float32)

    # Populate the cache with some data
    key_states = torch.randn(max_batch_size, 8, 1024, 64, dtype=torch.float32)
    value_states = torch.randn(max_batch_size, 8, 1024, 64, dtype=torch.float32)
    cache_manager.update(key_states, value_states,
                         layer_idx=0, cache_kwargs={"cache_position": torch.tensor([0,1,2,3])})

    # Check if the cache is empty
    print(f"Is cache empty? {cache_manager.is_empty(layer_idx=0)}")

    # Get non-empty and empty indices
    cache_status = cache_manager.get_non_empty_indices()
    print(f"Non-empty indices: {cache_status.non_empty}")
    print(f"Empty indices: {cache_status.empty}")

    # Reset cache for a specific sample
    cache_manager.reset_sample_cache(sample_index=0)

    # Check if the cache is empty after reset
    print(f"Is cache empty after reset? {cache_manager.is_empty(layer_idx=0)}")
