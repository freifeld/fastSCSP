#!/usr/bin/env python
"""
Created on Sun May 25 09:35:11 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

from pycuda.driver import Context

def init_the_device_if_needed(do_it_anyway=False):
    if do_it_anyway:
        print 'import pycuda.autoinit'
        import pycuda.autoinit
        return
    try:
        Context.get_device()
    except:
        # Presumably, the line above failed because of something like that:
        # "LogicError: cuCtxGetDevice failed: not initialized"
        # -- initialize the device
        print 'import pycuda.autoinit'
        import pycuda.autoinit


if __name__ == "__main__":
#    init_the_device_if_needed(do_it_anyway=True)
    init_the_device_if_needed()