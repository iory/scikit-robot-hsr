# flake8: noqa

import pkg_resources


__version__ = pkg_resources.get_distribution('scikit-robot-hsr').version


from skrobot_hsr.hsrb import HSRB
