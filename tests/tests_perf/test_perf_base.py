#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import os
import statistics
import time

import dpnp
import numpy

class DPNPTestPerfBase:
    seed = 777
    repeat=15
    sep = ":"
    results_data = dict()
    print_width = [10, 8, 6, 10]
    print_num_width = 10
    
    @classmethod
    def setup_class(cls):
        cls.results_data.clear()


    @classmethod
    def teardown_class(cls):
        cls.print_csv(cls)


    def add(self, name, lib, dtype, size, result):
        """Add performance testing results into global storage."""

        # Python does not automatically create a dictionary when you use multilevel keys
        if not self.results_data.get(name, False):
            self.results_data[name] = dict()

        if not self.results_data[name].get(dtype, False):
            self.results_data[name][dtype] = dict()

        if not self.results_data[name][dtype].get(lib, False):
            self.results_data[name][dtype][lib] = dict()

        self.results_data[name][dtype][lib][size] = result


    def dpnp_benchmark(self, name, lib, dtype, size, *args, **kwargs):
        """
        Test performance of specified function.

        Parameters
        ----------
        name : str
            name of the function
        lib : type
            library
        dtype : dtype
            data type of the input array
        size : int
            size of the input array
        args : tuple
            position parameters of the function
        kwargs : dict
            key word parameters of the function
        """
        examine_function = getattr(lib, name)

        exec_times = []
        for iteration in range(self.repeat):
            start_time = time.perf_counter()
            result = examine_function(*args, **kwargs)
            end_time = time.perf_counter()

            exec_times.append(end_time - start_time)

        self.add(name, lib, dtype, size, exec_times)

    
    def print_head(self):
        print()
        pw = self.print_width
        pwn = self.print_num_width
        print(f"Function".center(pw[0]), end=self.sep)
        print(f"type".center(pw[1]), end=self.sep)
        print(f"lib".center(pw[2]), end=self.sep)
        print(f"size".center(pw[3]), end=self.sep)
        print(f"median".center(pwn), end=self.sep)
        print(f"max".center(pwn), end=self.sep)
        print(f"min".center(pwn), end=self.sep)
        print()

    def print_csv(self):
        """Print performance testing results from global data storage."""
        self.print_head(self)
        pw = self.print_width
        pwn = self.print_num_width

        for func_name, func_results in self.results_data.items():
            for dtype_id, dtype_results in func_results.items():
                dtype_id_prn = dtype_id.__name__
                graph_data = dict()
                for lib_id, lib_results in dtype_results.items():
                    lib_id_prn = lib_id.__name__
                    
                    graph_data[lib_id_prn] = {"x": list(), "y": list()}
                    for size, size_results in lib_results.items():
                        print(f"{func_name:{pw[0]}}", end=self.sep)
                        print(f"{dtype_id_prn:{pw[1]}}", end=self.sep)
                        print(f"{lib_id_prn:{pw[2]}}", end=self.sep)
                        print(f"{size:{pw[3]}}", end=self.sep)

                        val_min = min(size_results)
                        val_max = max(size_results)
                        val_median = statistics.median(size_results)

                        print(f"{val_median:{pwn}.2e}", end=self.sep)
                        print(f"{val_min:{pwn}.2e}", end=self.sep)
                        print(f"{val_max:{pwn}.2e}", end=self.sep)
                        
                        print()

                        # prepare data for graphs
                        graph_data[lib_id_prn]["x"].append(size)
                        graph_data[lib_id_prn]["y"].append(val_median)

                self.plot_graph(self, graph_data, func_name=func_name, lib=lib_id_prn, type=dtype_id_prn)

    def plot_graph(self, graph_data, func_name, lib, type):
        """Plot graph with testing results from global data storage."""
        import matplotlib.pyplot as plt
        
        plt.suptitle(f"'{func_name}' time in (s)");
        plt.title(f"for '{type}' data type");
        plt.xlabel("number of elements")
        plt.ylabel("time(s)")
       
        for lib_id, axis in graph_data.items():
            plt.plot(axis["x"], axis["y"], label=lib_id, marker='.')

        plt.legend()
        plt.tight_layout()

        plt.savefig("dpnp_perf_" + func_name + "_" + type + ".jpg", dpi=300)
        plt.close()
