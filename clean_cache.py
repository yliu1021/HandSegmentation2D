from __future__ import print_function
import os
import sys
import shutil


def clean_cache():
    base = "/Users/Yuhan/Documents/DocumentsMacBook/Programming/_Projects/Experimental/HandTracking/_DataSets/RHD_published_v2/cached/"
    base += "evaluation/"

    caches = os.listdir(base)
    caches = [base+x for x in caches if os.path.isdir(base+x)]
    caches.sort()

    for i, cache in enumerate(caches, 1):
        sys.stdout.write("\r")
        sys.stdout.flush()
        print("Deleting: %d/%d\t%f" % (i, len(caches), float(i)/len(caches)), end="")
        files = os.listdir(cache)
        files_to_delete = [os.path.join(cache, x) for x in files if not (x.startswith("input") or x.startswith("mask"))]
        for file_to_delete in files_to_delete:
            os.remove(file_to_delete)


def clean_saved_graphs(generation):
    base = "/Users/Yuhan/Documents/DocumentsMacBook/Programming/_Projects/Experimental/AI/EvolutionaryNN/networks/"
    base += "generation_%d/" % generation

    organisms = os.listdir(base)
    organisms = [base+x for x in organisms if os.path.isdir(base+x)]
    organisms.sort()

    for i, organism in enumerate(organisms, 1):
        sys.stdout.write("\r")
        sys.stdout.flush()
        print("Deleting: %d/%d\t%d%%" % (i, len(organisms), 100.0*float(i)/len(organisms)), end="")
        file_to_delete = os.path.join(organism, "saved_graphs")
        shutil.rmtree(file_to_delete)


clean_saved_graphs(9)
