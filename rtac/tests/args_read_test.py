import unittest
import sys
sys.path.append('rtac')
from utils.read_io import read_args
from ac_functionalities.rtac_data import ACMethod


class TestArgsRead(unittest.TestCase):
    default = {'verbosity': 0, 'wrapper': 'No wrapper chosen!',
               'wrapper_name': 'No wrapper class name.', 
               'timeout': 300, 'contenders': 30, 'keeptop': 4, 'pws': False,
               'usedata': None, 'experimental': False, 'chance': 25,
               'mutate': 10, 'kill': 5,
               'feature_gen': '', 'nc_pca_f': 3,
               'nc_pca_p': 5, 'jfm': 'polynomial', 'omega': 1.0, 'gamma': 1,
               'alpha': 0.2, 'ac': ACMethod(1), 'paramlimit': 100000,
               'baselineperf': False, 'cpplt': 2,
               'log_folder': 'logs',
               'param_file': 'No parameter file given!', 'resume': False}

    scen_file = {'verbosity': 1, 'wrapper': 'wrapper.lacidac',
                 'wrapper_name': 'Lacidac', 'timeout': 3, 'contenders': 50,
                 'keeptop': 3, 'pws': True, 'usedata': 'y',
                 'experimental': True, 'chance': 2, 'mutate': 100,
                 'kill': 50.0,
                 'feature_gen': 'feature_gen/lacidac_feats.py',
                 'nc_pca_f': 33, 'nc_pca_p': 55, 'jfm': 'linear',
                 'omega': 1e-06, 'gamma': 10.0, 'alpha': 0.02,
                 'ac': ACMethod(1), 'paramlimit': 1100000.0,
                 'baselineperf': True, 'cpplt': 89,
                 'log_folder': 'logs',
                 'param_file': 'rtac/tests/test_data/params_lacidac.json',
                 'resume': False}

    def test_default_args(self, default=default):
        scenario = read_args()

        for key in default:
            # print(scenario.__dict__[key])
            print('\n', key)
            self.assertEqual(scenario.__dict__[key], default[key])

    def test_scenario(self, scen_file=scen_file):
        scenario = read_args('rtac/tests/test_data/scenario.txt')

        for key in scen_file:
            print(scenario.__dict__[key])
            self.assertEqual(scenario.__dict__[key], scen_file[key])

    def test_scenario_sysargs(self):
        sys.argv.extend(['main.py', '--contenders', '5'])
        scenario = read_args('rtac/tests/test_data/scenario.txt',
                             sys.argv)

        print(scenario.contenders)

        self.assertEqual(scenario.contenders, 5)


if __name__ == '__main__':
    unittest.main()
