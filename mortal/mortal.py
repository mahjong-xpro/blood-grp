import prelude

import os
import sys
import json
import torch
from datetime import datetime, timezone
from model import Brain, DQN
from engine import MortalEngine
from common import filtered_trimmed_lines
from libblood.mjai import Bot
# from libblood.dataset import Grp
from config import config

USAGE = '''Usage: python mortal.py <ID>

ARGS:
    <ID>    The player ID, an integer within [0, 3].'''

def main():
    try:
        player_id = int(sys.argv[-1])
        assert player_id in range(4)
    except:
        print(USAGE, file=sys.stderr)
        sys.exit(1)
    review_mode = os.environ.get('MORTAL_REVIEW_MODE', '0') == '1'

    device = torch.device('cpu')
    state = torch.load(config['control']['state_file'], weights_only=True, map_location=torch.device('cpu'))
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    num_blocks = cfg['resnet']['num_blocks']
    conv_channels = cfg['resnet']['conv_channels']
    if 'tag' in state:
        tag = state['tag']
    else:
        time = datetime.fromtimestamp(state['timestamp'], tz=timezone.utc).strftime('%y%m%d%H')
        tag = f'mortal{version}-b{num_blocks}c{conv_channels}-t{time}'

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])

    engine = MortalEngine(
        mortal,
        dqn,
        version = version,
        is_oracle = False,
        device = device,
        enable_amp = False,
        enable_quick_eval = not review_mode,
        enable_rule_based_agari_guard = True,
        name = 'mortal',
    )
    bot = Bot(engine, player_id)

    if review_mode:
        logs = []
    for line in filtered_trimmed_lines(sys.stdin):
        if review_mode:
            logs.append(line)

        if reaction := bot.react(line):
            print(reaction, flush=True)
        elif review_mode:
            print('{"type":"none","meta":{"mask_bits":0}}', flush=True)

    if review_mode:
        # GRP (Win Probability) calculation is removed for Bloody Battle
        # print(json.dumps(extra_data), flush=True) 
        pass

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
