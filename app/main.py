import json
import os
import random
import bottle


import json
from bottle import HTTPResponse

import numpy as np
import tensorflow as tf
from stable_baselines import A2C

#import matplotlib.pyplot as plt


# Import Snake Environment
import sys
sys.path.insert(0, "app/app/")
#
# Make Snake Environment
BOARD_SIZE=11

# Import Policy
model = A2C.load("yeeters.zip")

# Direction dictionary
dir_dict = {0:'right', 1:'down', 2:'left', 3:'up'}

# Blocks
NB = 0 # Null Block
FB = 2 # Food Block
HB = 3 # Head block
BB = 1
AN = np.array([0,0,0,0,0,0,0])


x_tail_before = None
y_tail_before = None
turn =0


def make_move(data):
    global turn

	# Create Board
    state = np.array([([[NB]*(7)]*(BOARD_SIZE))\
                         for i in range((BOARD_SIZE))], dtype=np.int32)

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
            state[y][x][0] = HB		
        else:
            state[y][x][0] = BB
	
	# Add Enemy Snakes
    master_id = data['you']['id']
    snakes = data['board']['snakes']
    j=0
    for snake in snakes:
        if snake['id']!=master_id:
            j+=1
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
                    state[y][x][j] = HB
                else:
                    state[y][x][j] = BB
	
	
	# Add fooda
    food_x_coords = [d['x'] for d in data['board']['food']]
    food_y_coords = [d['y'] for d in data['board']['food']]
    for i, (x, y) in enumerate(zip(food_x_coords, food_y_coords)):
        state[y][x][4] = FB
    
    
    # Compute longest Snake Stuff
    master_head_x, master_head_y = data['you']['body'][0]
    master_length = len(data['you']['body'])
    for snake in snakes:
        if snake['id']!=master_id:
            head_x, head_y = snake['body'][0]['x'], snake['body'][0]['y']
            length = len(snake['body'])
            if length>=master_length:
                state[head_y][head_x][5]=1
            elif length<master_length:
                state[head_y][head_x][6]=1
    
    turn+=1
    
    action = model.predict(70*state, deterministic=True)[0]
    if move_snake(data, state, action):
        return dir_dict[action]
    else:
        actions = [0,1,2,3]
        alives = [move_snake(data, state, action) for action in actions]
        indices = [i for i, x in enumerate(alives) if x == True]
        if len(indices)>0:
            return dir_dict[random.choice([actions[i] for i in indices])]
        else:
            return dir_dict[0]

def move_snake(data, state, action):
        global turn
        print(turn)
        # Returns alive
        
        old_position_x = data['you']['body'][0]['x']
        old_position_y = data['you']['body'][0]['y']
        # Using curr position and action find out which block snake has landed on 
        if action == 0: (new_y, new_x) = (old_position_y, old_position_x+1)
        elif action == 1: (new_y, new_x) = (old_position_y+1, old_position_x)
        elif action == 2: (new_y, new_x) = (old_position_y, old_position_x-1)
        elif action == 3: (new_y, new_x) = (old_position_y-1, old_position_x)
        
        # If landed out of bounds then necessarily dead
        if new_x<0 or new_y<0 or new_x>=BOARD_SIZE or new_y>=BOARD_SIZE:
            return False
        # If landed on food block then give reward and set just eaten
        if state[new_y][new_x][4] == FB:
            return True
       
       # If landed on a tail block and the corresponding snake has NOT eaten then fine
        curr_tail_x = data['you']['body'][-1]['x']
        curr_tail_y = data['you']['body'][-1]['y']
        if new_y==curr_tail_y and new_x==curr_tail_x:
            print('landed on tail on turn {}'.format(turn))
            if data['you']['health']==100 or len(data['you']['body'])<4:
                return False
            else:
                return True
        

        if np.array_equal(state[new_y][new_x], AN):
            return True
        else:
            return False

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
