# AMBER vacuum regression fixture

These unmodified 22-atom alanine-dipeptide files come from the OpenMM Python
API test suite at commit `ad0c51cf041081c138507e23cdbf8360d983ac4e`:

- https://github.com/openmm/openmm/blob/ad0c51cf041081c138507e23cdbf8360d983ac4e/wrappers/python/tests/systems/alanine-dipeptide-implicit.prmtop
- https://github.com/openmm/openmm/blob/ad0c51cf041081c138507e23cdbf8360d983ac4e/wrappers/python/tests/systems/alanine-dipeptide-implicit.inpcrd

SHA256, preserving the original bytes:

- prmtop: `2ce81216c7e18fd4d354fac44e22ba3843d89e297884bd6389a4cd57c74ecf6e`
- inpcrd: `b8a151fd35b909de52b7f50b09f4a0c9fc5221d0b423c6a9ec3133bf867c4954`

The test imports these parameters as vacuum NoCutoff and explicitly moves
particles 6 through 15. This checks parameter import, physical energies and
forces against OpenMM Reference; it is not a protein-ligand docking benchmark.
The filename's `implicit` designation does not enable implicit solvent in the
Torch potential.

The OpenMM API and its Reference platform are distributed under the MIT terms
in the upstream [license notice](https://github.com/openmm/openmm/blob/ad0c51cf041081c138507e23cdbf8360d983ac4e/docs-source/licenses/Licenses.txt).
The corresponding copyright and permission notice follows:

Portions copyright © 2008-2025 Stanford University and the Authors.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject
to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
