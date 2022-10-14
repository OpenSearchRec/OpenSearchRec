from OpenSearchRec.post_ranking import merge_lists


def test_merge_lists():
    assert merge_lists([[1, 2, 3], ["a", "b", "c"]]) == [1, 'a', 2, 'b', 3, 'c']
    assert merge_lists([[1, 2, 3, 4, 5, 6, 7, 8, 9], ["a", "b", "c"]], [2, 1]) == [1, 2, 'a', 3, 4, 'b', 5, 6, 'c', 7, 8, 9]
    assert merge_lists([[1, 2], ["1", "2"], ["a", "b"]])
    assert merge_lists([[1, 2], [2, 3]]) == [1, 2, 2, 3]
    assert merge_lists([[1, 2], [2, 3]], remove_duplicates=True) == [1, 2, 3]
    assert merge_lists([[1, 2], ["2", "3"]], remove_duplicates=True) == [1, "2", 2, "3"]
    assert merge_lists([[1, 2], ["2", "3"]], remove_duplicates=True, deduplication_key_function=lambda i: str(i)) == [1, "2", "3"]
