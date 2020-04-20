import json
import os
import random
import bottle

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import json
from bottle import HTTPResponse

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec

#import matplotlib.pyplot as plt


# Import Snake Environment
import sys
sys.path.insert(0, "app/app/")

# Make Snake Environment
tf.random.set_seed(888)
BOARD_SIZE=11

# Import Policy
policy = tf.compat.v2.saved_model.load('policy_8')

# Direction dictionary
dir_dict = {0:'right', 1:'down', 2:'left', 3:'up'}

# Blocks
NB = 0 # Null Block
FB = 1 # Food Block
# All blocks >2 is snake
# Snake
HB = 2 # Head block
UB = 3 # Up then up OR down then down
SB = 4 # Right then right OR Left then left
UL = 5 # Up then left OR right then down
UR = 6 #Up then right OR left then down
RU = 7 # Right then up OR down then left
DR = 8 # Down then right OR left then up
# Enemy
HB_en = 9 # Head block
UB_en = 10 # Up then up OR down then down
SB_en = 11 # Right then right OR Left then left
UL_en = 12 # Up then left OR right then down
UR_en = 13 #Up then right OR left then down
RU_en = 14 # Right then up OR down then left
DR_en = 15 # Down then right OR left then up

NB_obs = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]], dtype=np.int32)
FB_obs = np.array([[3,3,3,3,3], [3,3,3,3,3], [3,3,3,3,3], [3,3,3,3,3], [3,3,3,3,3]], dtype=np.int32)
# Master
HB_obs = np.array([[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]], dtype=np.int32)
SB_obs = np.array([[0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0]], dtype=np.int32)
UB_obs = np.array([[0,0,0,0,0], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [0,0,0,0,0]], dtype=np.int32)
DR_obs = np.array([[0,0,0,0,0], [1,1,1,1,0], [1,1,1,1,0], [1,1,1,1,0], [0,1,1,1,0]], dtype=np.int32)
UR_obs = np.array([[0,0,0,0,0], [0,1,1,1,1], [0,1,1,1,1], [0,1,1,1,1], [0,1,1,1,0]], dtype=np.int32)
RU_obs = np.array([[0,1,1,1,0], [1,1,1,1,0], [1,1,1,1,0], [1,1,1,1,0], [0,0,0,0,0]], dtype=np.int32)
UL_obs = np.array([[0,1,1,1,0], [0,1,1,1,1], [0,1,1,1,1], [0,1,1,1,1], [0,0,0,0,0]], dtype=np.int32)
# Enemies
HB_en_obs = 2*np.array([[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]], dtype=np.int32)
SB_en_obs = 2*np.array([[0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0]], dtype=np.int32)
UB_en_obs = 2*np.array([[0,0,0,0,0], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [0,0,0,0,0]], dtype=np.int32)
DR_en_obs = 2*np.array([[0,0,0,0,0], [1,1,1,1,0], [1,1,1,1,0], [1,1,1,1,0], [0,1,1,1,0]], dtype=np.int32)
UR_en_obs = 2*np.array([[0,0,0,0,0], [0,1,1,1,1], [0,1,1,1,1], [0,1,1,1,1], [0,1,1,1,0]], dtype=np.int32)
RU_en_obs = 2*np.array([[0,1,1,1,0], [1,1,1,1,0], [1,1,1,1,0], [1,1,1,1,0], [0,0,0,0,0]], dtype=np.int32)
UL_en_obs = 2*np.array([[0,1,1,1,0], [0,1,1,1,1], [0,1,1,1,1], [0,1,1,1,1], [0,0,0,0,0]], dtype=np.int32)

return_state = np.array([([NB_obs]*BOARD_SIZE) for i in range(BOARD_SIZE)], dtype=np.int32)

x_tail_before = None
y_tail_before = None
turn =0

def get_block_player(dx1, dy1, dx2, dy2):
	arr = (dx1, -dy1, dx2, -dy2)
	if (arr==(0,1,0,1) or arr==(0,-1,0,-1)):
		return UB
	elif (arr==(1,0,1,0) or arr==(-1,0,-1,0)):
		return SB
	elif (arr==(0,1,-1,0) or arr==(1,0,0,-1)):
		return UL
	elif (arr==(0,1,1,0) or arr==(-1,0,0,-1)):
		return UR
	elif (arr==(1,0,0,1) or arr==(0,-1,-1,0)):
		return RU
	elif (arr==(0,-1,1,0) or arr==(-1,0,0,1)):
		return DR
	else:
		print('arr not found')
		print(arr)
		
def get_block_enemy(dx1, dy1, dx2, dy2):
	arr = (dx1, -dy1, dx2, -dy2)
	if (arr==(0,1,0,1) or arr==(0,-1,0,-1)):
		return UB_en
	elif (arr==(1,0,1,0) or arr==(-1,0,-1,0)):
		return SB_en
	elif (arr==(0,1,-1,0) or arr==(1,0,0,-1)):
		return UL_en
	elif (arr==(0,1,1,0) or arr==(-1,0,0,-1)):
		return UR_en
	elif (arr==(1,0,0,1) or arr==(0,-1,-1,0)):
		return RU_en
	elif (arr==(0,-1,1,0) or arr==(-1,0,0,1)):
		return DR_en
	else:
		print('arr not found')
		print(arr)

def get_board(state):
	ARR=np.copy(return_state)
	ARR[state==NB]=NB_obs
	ARR[state==FB]=FB_obs
	ARR[state==HB]=HB_obs
	ARR[state==UB]=UB_obs
	ARR[state==SB]=SB_obs
	ARR[state==UL]=UL_obs
	ARR[state==UR]=UR_obs
	ARR[state==RU]=RU_obs
	ARR[state==DR]=DR_obs
	ARR[state==HB_en]=HB_en_obs
	ARR[state==UB_en]=UB_en_obs
	ARR[state==SB_en]=SB_en_obs
	ARR[state==UL_en]=UL_en_obs
	ARR[state==UR_en]=UR_en_obs
	ARR[state==RU_en]=RU_en_obs
	ARR[state==DR_en]=DR_en_obs	
	return np.expand_dims(ARR.transpose((0,3,1,2)).reshape(5*BOARD_SIZE, 5*BOARD_SIZE), axis=-1)

def make_move(data):
	global x_tail_before
	global y_tail_before
	global turn


	# Create Board
	state = np.array([([NB]*BOARD_SIZE) for i in range(BOARD_SIZE)], dtype=np.int32)

	# Add your snake
	if turn==0:
		master_x_coords = [d['x'] for d in data['you']['body']][0:1]
		master_y_coords = [d['y'] for d in data['you']['body']][0:1]
	elif turn==1:
		master_x_coords = [d['x'] for d in data['you']['body']][0:2]
		master_y_coords = [d['y'] for d in data['you']['body']][0:2]
	else:
		master_x_coords = [d['x'] for d in data['you']['body']]
		master_y_coords = [d['y'] for d in data['you']['body']]
	for i, (x, y) in enumerate(zip(master_x_coords, master_y_coords)):
		if i == 0:
			state[y][x] = HB
		elif i==len(master_x_coords)-1:
			x_prev = master_x_coords[i-1]
			y_prev = master_y_coords[i-1]
			
			dx1 = x-x_prev
			dy1 = y-y_prev
			if dx1==1 or dx1==-1:
				state[y][x] = SB
			else:
				state[y][x] = UB
				
		else:
			x_next = master_x_coords[i+1]; x_prev = master_x_coords[i-1]
			y_next = master_y_coords[i+1]; y_prev = master_y_coords[i-1]
			
			if (x==x_prev and y==y_prev):
				continue
			if (x==x_next and y==y_next):
				continue
				
			dx1 = int(x-x_prev); dx2 = int(x_next-x)
			dy1 = int(y-y_prev); dy2 = int(y_next-y)
			state[y][x] = get_block_player(dx1, dy1, dx2, dy2)
	
	# Add Enemy Snakes
	master_id = data['you']['id']
	snakes = data['board']['snakes']
	for snake in snakes:
		if snake['id']!=master_id:
			if turn==0:
				x_coords = [d['x'] for d in snake['body']][0:1]
				y_coords = [d['y'] for d in snake['body']][0:1]
			elif turn==1:
				x_coords = [d['x'] for d in snake['body']][0:2]
				y_coords = [d['y'] for d in snake['body']][0:2]
			else:
				x_coords = [d['x'] for d in snake['body']]
				y_coords = [d['y'] for d in snake['body']]
			for i, (x, y) in enumerate(zip(x_coords, y_coords)):
				if i == 0:
					state[y][x] = HB_en
				elif i==len(x_coords)-1:
					x_prev = x_coords[i-1]
					y_prev = y_coords[i-1]
					
					dx1 = x-x_prev
					dy1 = y-y_prev
					if dx1==1 or dx1==-1:
						state[y][x] = SB_en
					else:
						state[y][x] = UB_en
						
				else:
					x_next = x_coords[i+1]; x_prev = x_coords[i-1]
					y_next = y_coords[i+1]; y_prev = y_coords[i-1]
					
					if (x==x_prev and y==y_prev):
						continue
					if (x==x_next and y==y_next):
						continue
						
					dx1 = int(x-x_prev); dx2 = int(x_next-x)
					dy1 = int(y-y_prev); dy2 = int(y_next-y)
					state[y][x] = get_block_enemy(dx1, dy1, dx2, dy2)
	
	
	# Add food
	food_x_coords = [d['x'] for d in data['board']['food']]
	food_y_coords = [d['y'] for d in data['board']['food']]
	for i, (x, y) in enumerate(zip(food_x_coords, food_y_coords)):
		state[y][x] = FB
	'''
	if turn%10==0:
		plt.imshow(np.squeeze(get_board(state), axis=-1))
		plt.savefig('test{}.png'.format(turn))
	action_step = policy.action(ts.restart(tf.convert_to_tensor([get_board(state)]), batch_size=1))
	'''
	turn+=1
	action_taken = take_action(data, state, action_step.action.numpy()[0])
	return dir_dict[action_taken]

def take_action(data, state, action):
	head_x = data['you']['body'][0]['x']
	head_y = data['you']['body'][0]['y']
        
	kill_itself = False
	possible_random_choices = [0,1,2,3]
	
	# If action would kill the snake then choose a list of the other three actions
	
	# Action 0
	if (head_x+1 == BOARD_SIZE):
		if (action==0):
			kill_itself = True
		possible_random_choices.remove(0)
	elif state[head_y][head_x+1]>2:
		if (action==0):
			kill_itself = True
		possible_random_choices.remove(0)
	
	# Action 1
	if (head_y+1 == BOARD_SIZE):
		if (action==1):
			kill_itself = True
		possible_random_choices.remove(1)
	elif state[head_y+1][head_x]>2:
		if (action==1):
			kill_itself = True
		possible_random_choices.remove(1)
	
	# Action 2
	if (head_x-1 == -1):
		if (action==2):
			kill_itself = True
		possible_random_choices.remove(2)
	elif state[head_y][head_x-1]>2:
		if (action==2):
			kill_itself = True
		possible_random_choices.remove(2)
		
	# Action 3
	if (head_y-1 == -1):
		if (action==3):
			kill_itself = True
		possible_random_choices.remove(3)
	elif state[head_y-1][head_x]>2:
		if (action==3):
			kill_itself = True
		possible_random_choices.remove(3)
	
	if kill_itself:
		if (len(possible_random_choices) == 0):
			return 0
		# Else pick one of the random actions
		else:
			a = random.choice(possible_random_choices)
			return a
	
	else:
		return action


@bottle.route('/')
def index():
    return '''
    Battlesnake documentation can be found at
       <a href="https://docs.battlesnake.com">https://docs.battlesnake.com</a>.
    '''


@bottle.route('/static/<path:path>')
def static(path):
    """
    Given a path, return the static file located relative
    to the static folder.

    This can be used to return the snake head URL in an API response.
    """
    return bottle.static_file(path, root='static/')


@bottle.post('/ping')
def ping():
    """
    A keep-alive endpoint used to prevent cloud application platforms,
    such as Heroku, from sleeping the application instance.
    """
    return ping_response()


@bottle.post('/start')
def start():
    data = bottle.request.json

    """
    TODO: If you intend to have a stateful snake AI,
            initialize your snake state here using the
            request's data if necessary.
    """
    print(json.dumps(data))

    color = "#00FF00"

    return start_response(color)


@bottle.post('/move')
def move():
    # Get data
    data_raw = bottle.request.json
    data = json.loads(json.dumps(data_raw))

    return move_response(make_move(data))


@bottle.post('/end')
def end():
	data = bottle.request.json
	
	
	"""
    TODO: If your snake AI was stateful,
        clean up any stateful objects here.
	"""
	print(json.dumps(data))

	return end_response()

def ping_response():
    return HTTPResponse(
        status=200
    )

def start_response(color):
    assert type(color) is str, \
        "Color value must be string"

    return HTTPResponse(
        status=200,
        headers={
            "Content-Type": "application/json"
        },
        body=json.dumps({
            "color": color
        })
    )

def move_response(move):
    assert move in ['up', 'down', 'left', 'right'], \
        "Move must be one of [up, down, left, right]"

    return HTTPResponse(
        status=200,
        headers={
            "Content-Type": "application/json"
        },
        body=json.dumps({
            "move": move
        })
    )

def end_response():
    return HTTPResponse(
        status=200
    )

# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == '__main__':
    
    # Add Action Policy
    # Create SnakeEnv
    
    bottle.run(
        application,
        host=os.getenv('IP', '0.0.0.0'),
        port=os.getenv('PORT', '8080'),
        debug=os.getenv('DEBUG', True)
    )
