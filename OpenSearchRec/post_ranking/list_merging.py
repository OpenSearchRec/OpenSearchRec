from typing import List, Optional


def merge_lists(
        lists: List[List],
        list_frequencies: Optional[List[int]] = None,
        remove_duplicates=False,
        deduplication_key_function=None):
    assert list_frequencies is None or (len(lists) == len(list_frequencies))
    if list_frequencies is not None:
        for freq in list_frequencies:
            assert freq > 0

    list_idxs = [0 for _ in lists]
    list_lengths = [len(list_) for list_ in lists]
    if list_frequencies is None:
        list_frequencies = [1 for _ in lists]
    else:
        list_frequencies = [f for f in list_frequencies]

    merged_list = []
    merged_list_key_values = set() # for deduplication
    while list_idxs != list_lengths:
        for l_idx, list_ in enumerate(lists):
            freq = list_frequencies[l_idx]
            num_added = 0
            while num_added < freq:
                if list_idxs[l_idx] == list_lengths[l_idx]:
                    break
                else:
                    next_element = list_[list_idxs[l_idx]]
                    if not remove_duplicates:
                        merged_list.append(next_element)
                        num_added += 1
                    else:
                        if deduplication_key_function is None:
                            key = next_element
                        else:
                            key = deduplication_key_function(next_element)
                        if key not in merged_list_key_values:
                            merged_list.append(next_element)
                            num_added += 1
                            merged_list_key_values.add(key)

                    list_idxs[l_idx] += 1

    return merged_list
