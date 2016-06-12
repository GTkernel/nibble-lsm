// What:    C interface for linking to libcukcoo
// Why:     Rust doesn't link directly with C++, currently.
// Desc:    Make one instance of the cuckoo hash table and provide
//          pass-through functions to each of the instance members.
// Use of this interface should only be done via the provided Rust
// wrapper for safety.  This file compiles to a library that
// includes libcityhash.a The safe Rust interface will link against
// this derivative library which exports the below C interface, hiding
// the libcuckoo C++ interface. The file build.rs will handle
// compiling us.
//
// Test this code compiles correctly with
// g++ cuckoo.cc -c -o cuckoo.o \
//          -std=c++11 -O3 -Ithirdparty/libcuckoo/src
// If any changes need to be made, update this comment, then reflect
// them back to build.rs

#include <iostream>
#include <cstdlib>
#include <mutex>
#include <cuckoohash_map.hh>

extern "C" {
    typedef char* KType;
    typedef uint64_t VType;

    // Singleton instance of the hash table.
    static cuckoohash_map<KType, VType> *cuckoomap = nullptr;

    // Lock used only for initialization.
    static std::mutex singleton_lock;

    void libcuckoo_init(void) {
        if (!cuckoomap && singleton_lock.try_lock()) {
            cuckoomap = new cuckoohash_map<KType,VType>();
            if (!cuckoomap) {
                std::cerr << "Error: OOM allocating cuckoo table"
                    << std::endl;
                std::cerr.flush();
                abort();
            }
            singleton_lock.unlock();
        }
        // if multiple threads try to call init, and we did not get
        // the lock, wait for the winning thread to complete
        // initialization
        while (!cuckoomap)
            ;
    }

    void libcuckoo_clear(void) {
        assert(cuckoomap);
        cuckoomap->clear();
    }

    size_t libcuckoo_size(void) {
        assert(cuckoomap);
        return cuckoomap->size();
    }

    bool libcuckoo_empty(void) {
        assert(cuckoomap);
        return cuckoomap->empty();
    }

    bool libcuckoo_find(const KType &key, VType &value) {
        assert(cuckoomap);
        return cuckoomap->find(key, value);
    }

    bool libcuckoo_contains(const KType &key) {
        assert(cuckoomap);
        return cuckoomap->contains(key);
    }


    bool libcuckoo_insert(const KType &key, VType &value) {
        assert(cuckoomap);
        try {
            return cuckoomap->insert(key, value);
        }
        // FIXME do something more meaningful with these
        catch (libcuckoo_load_factor_too_low &e) {
            std::cerr << "Error: cuckoo load factor too low"
                << std::endl;
            std::cerr.flush();
            abort();
        }
        catch (libcuckoo_maximum_hashpower_exceeded &e) {
            std::cerr << "Error: cuckoo load max hashpower exceeded"
                << std::endl;
            std::cerr.flush();
            abort();
        }
        return false; // currently not reached
    }

    bool libcuckoo_erase(const KType &key) {
        assert(cuckoomap);
        return cuckoomap->erase(key);
    }

    bool libcuckoo_update(const KType &key, VType &value) {
        assert(cuckoomap);
        return cuckoomap->update(key, value);
    }

    bool libcuckoo_reserve(size_t n) {
        assert(cuckoomap);
        try {
            return cuckoomap->reserve(n);
        }
        // FIXME do something more meaningful with these
        catch (libcuckoo_maximum_hashpower_exceeded &e) {
            std::cerr << "Error: cuckoo load max hashpower exceeded"
                << std::endl;
            std::cerr.flush();
            abort();
        }
    }
}
