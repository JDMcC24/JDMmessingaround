from itertools import combinations

def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    l = len(A)
    pset = []
    for n in range(1,l+1):
        new_sets = list(combinations(A,n))
        for s in new_sets:
            pset.append(s)
    pset.append(set(A))
    pset.append(set())
    return pset

print(power_set({'a','b','c'}))