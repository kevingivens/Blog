
cdef extern from "/home/givenskm/Projects/pycparser/examples/c_files/trie.h":

    ctypedef struct Trie:
        pass

    ctypedef void *TrieValue
    Trie *trie_new()
    void trie_free(Trie *trie)
    int trie_insert(Trie *trie, char *key, TrieValue value)
    int trie_insert_binary(Trie *trie, unsigned char *key, int key_length, TrieValue value)
    TrieValue trie_lookup(Trie *trie, char *key)
    TrieValue trie_lookup_binary(Trie *trie, unsigned char *key, int key_length)
    int trie_remove(Trie *trie, char *key)
    int trie_remove_binary(Trie *trie, unsigned char *key, int key_length)
    unsigned int trie_num_entries(Trie *trie)

