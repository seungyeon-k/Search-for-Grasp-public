import argparse
from omegaconf import OmegaConf
from control.controller import Controller
import ast
import time

def run(cfg):

	# get settings
	device = cfg.device
	enable_gui = cfg.get("enable_gui", False)
	idx = cfg.idx

	# control
	controller = Controller(
		cfg.controller, 
		device=device,
		enable_gui=enable_gui,
		)
	controller.control(exp_idx=idx)
	time.sleep(10)

def parse_arg_type(val):
	if val[0] == '[' and val[-1] == ']':
		return ast.literal_eval(val)
	if val.isnumeric():
		return int(val)
	if (val == 'True') or (val == 'true'):
		return True
	if (val == 'False') or (val == 'false'):
		return False
	if (val == 'None'):
		return None
	try:
		return float(val)
	except:
		return str(val)

def parse_unknown_args(l_args):
	"""convert the list of unknown args into dict
	this does similar stuff to OmegaConf.from_cli()
	I may have invented the wheel again..."""
	n_args = len(l_args) // 2
	kwargs = {}
	for i_args in range(n_args):
		key = l_args[i_args*2]
		val = l_args[i_args*2 + 1]
		assert '=' not in key, 'optional arguments should be separated by space'
		kwargs[key.strip('-')] = parse_arg_type(val)
	return kwargs


def parse_nested_args(d_cmd_cfg):
	"""produce a nested dictionary by parsing dot-separated keys
	e.g. {key1.key2 : 1}  --> {key1: {key2: 1}}"""
	d_new_cfg = {}
	for key, val in d_cmd_cfg.items():
		l_key = key.split('.')
		d = d_new_cfg
		for i_key, each_key in enumerate(l_key):
			if i_key == len(l_key) - 1:
				d[each_key] = val
			else:
				if each_key not in d:
					d[each_key] = {}
				d = d[each_key]
	return d_new_cfg

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str)
	parser.add_argument('--device', default=0)
	parser.add_argument('--debug', action='store_true')

	# process cfg
	args, unknown = parser.parse_known_args()
	d_cmd_cfg = parse_unknown_args(unknown)
	d_cmd_cfg = parse_nested_args(d_cmd_cfg)
	cfg = OmegaConf.load(args.config)
	cfg = OmegaConf.merge(cfg, d_cmd_cfg)

	# set device
	if args.device == 'cpu':
		cfg.device = 'cpu'
	elif args.device == 'any':
		cfg.device = 'cuda'
	else:
		cfg.device = f'cuda:{args.device}'

	# run
	run(cfg)

