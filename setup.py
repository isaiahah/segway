#!/usr/bin/env python

"""gmseg: DESCRIPTION

LONG_DESCRIPTION
"""

__version__ = "0.1.0a1"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

import sys

assert sys.version_info >= (2, 5)

from ez_setup import use_setuptools
use_setuptools()

from setuptools import find_packages, setup

doclines = __doc__.splitlines()
name, short_description = doclines[0].split(": ")
long_description = "\n".join(doclines[2:])

url = "http://noble.gs.washington.edu/~mmh1/software/%s/" % name.lower()
download_url = "%s%s-%s.tar.gz" % (url, name, __version__)

# XXX: remove these when the upstream packages are updated to fix these issues
dependency_links = ["http://pypi.python.org/packages/source/p/path.py/path-2.2.zip"]
# http://pytables.org/svn/pytables/branches/std-2.0#egg=tables-2.0.5.dev

classifiers = ["Natural Language :: English",
               "Programming Language :: Python"]

entry_points = """
[console_scripts]
gmseg = gmseg.run:main
gmsegimport = gmseg.importdata:main
gmsegimportseq = gmseg.importseq:main
"""

# XXX: warn: make sure you have LDFLAGS unset if you are building numpy

# need optbuild>0.1.2 for OptionBuilder_ShortOptWithSpace_TF
# need tables>2.04 (>=r3761) because there is a CArray fill bug until then
install_requires = ["textinput", "optbuild>0.1.2", "tables>2.0.4", "numpy", "path"]

setup(name=name,
      version=__version__,
      description=short_description,
      author="Michael Hoffman",
      author_email="mmh1@washington.edu",
      url=url,
      download_url=download_url,
      classifiers=classifiers,
      long_description=long_description,
      dependency_links=dependency_links,
      install_requires=install_requires,
      zip_safe=True,

      # XXX: this should be based off of __file__ instead
      packages=find_packages("."),
      include_package_data=True,
      entry_points=entry_points
      )
