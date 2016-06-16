// Test code for cuckoo C++ interface
// Have to build/run manually (can't use cargo)
// g++ cuckoo.cc -o cuckoo -std=c++11
//      -I../thirdparty/libcuckoo/src

#include <iostream>
#include <cstdlib>
#include <functional>

#include <assert.h>
#include <string.h>
#include <stdio.h>

// Bad style, but it's just for testing ;)
// Pulls in CStringHasher and CStringEqual
#include "../src/cuckoo.cc"

using namespace std;

// use the cuckoo API directly
// since we are using char*, we must provide a custom equal_to
// predicate, else the hashtable will be comparing pointers
void test_direct() {
    typedef char* KType;
    typedef uint64_t VType;
    CStringEqual eq;
    cuckoohash_map<KType, VType, CStringHasher, CStringEqual> *map;
    map = new cuckoohash_map<KType, VType, CStringHasher, CStringEqual>();
    assert( map );
    map->clear();
    assert( map->size() == 0 );
    assert( map->empty() == true );

    KType key = "hello", key2 = "na";
    VType v = 42, old;

    // same as key but different memory buffer...
    KType k3 = (KType) calloc(10, sizeof(*k3));
    assert(k3);
    strncpy(k3, key, strlen(key));

    //assert( map->contains(key2) == false );
    //assert( map->contains(key) == false );
    //assert( map->contains(k3) == false );

    assert( map->insert(key, v) == true );
    assert( map->insert(key, v) == false );
    assert( map->contains(key) == true );
    // these should behave as if using key
    assert( map->contains(k3) == true );
    assert( map->insert(k3, v) == false );

    assert( map->contains(key2) == false );

    assert( map->contains(key) == true );
    assert( map->find(key) == v );
    assert( map->erase2(key, &old) == true );
    assert( v == old );

    old = 0;
    assert( map->update2(key, v, old) == false );
    assert( old == 0 );
    assert( map->insert(key, 83) == true );
    assert( map->update2(key, v, old) == true );
    assert( old == 83 );

    old = 0;
    assert( map->erase2(k3, &old) == true );
    assert( old == v );
    assert( map->erase2(key, &old) == false );
}

// go through the C shim I created
void test_interface() {
    libcuckoo_init();
    assert( libcuckoo_size() == 0 );
    assert( libcuckoo_empty() == true );

    KType key = "hello", key2 = "na";
    VType v = 42ul, old;

    assert( libcuckoo_contains(key2) == false );
    assert( libcuckoo_contains(key) == false );

    assert( libcuckoo_insert(key, v) == true );
    assert( libcuckoo_insert(key, v) == false );
    assert( libcuckoo_contains(key) == true );
    assert( libcuckoo_contains(key2) == false );

    assert( libcuckoo_find(key, old) == true );
    assert( old == v );
    assert( libcuckoo_erase(key, old) == true );
    assert( v == old );

    old = 0;
    assert( libcuckoo_update(key, 83ul, old) == false );
    assert( old == 0ul );
    assert( libcuckoo_insert(key, old) == false );
    assert( libcuckoo_update(key, 0xbul, old) == true );
    assert( old == 83ul );

    old = 0ul;
    assert( libcuckoo_erase(key, old) == true );
    assert( old == 0xbul );
    old = 0ul;
    assert( libcuckoo_erase(key, old) == false );
    assert( old == 0ul );
}

int main() {
    test_direct();
    cout << "passed direct interface" << endl;
    test_interface();
    cout << "passed shim interface" << endl;
    return 0;
}
