from pkg_resources import DistributionNotFound, get_distribution

__all__ = ['RNN', 'utils']
try:
    __version__ = get_distribution('rnnsim').version
except DistributionNotFound:
    __version__ = '(local)'
