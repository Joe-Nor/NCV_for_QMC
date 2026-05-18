FC = /usr/bin/gfortran
FFLAGS = -O2 -g -fcheck=all -fbacktrace -Wall -Wextra -std=legacy -shared -fPIC

all: parity_prefix_lib.so parity_prefix_candidates_lib.so

parity_prefix_lib.so: parity_prefix_lib.f90
	$(FC) $(FFLAGS) -o $@ $<

parity_prefix_candidates_lib.so: parity_prefix_candidates_lib.f90
	$(FC) $(FFLAGS) -o $@ $<

clean:
	rm -f parity_prefix_lib.so parity_prefix_candidates_lib.so *.mod
