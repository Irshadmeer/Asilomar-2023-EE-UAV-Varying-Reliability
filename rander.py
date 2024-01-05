import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Render:
    def __init__(self):
        pass

    def animate_locations_list(self, info_list):
        # Loop through each dictionary in the list and animate the user and base station locations
        for info in info_list:
            # Extract information from the dictionary
            user_locations = info["user_location"]
            bs_locations = info["bs_location"]
            time_slot = info["time_slot"]

            fig, ax = plt.subplots()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(0,self.max_coordinates[0])
            ax.set_ylim(0,self.max_coordinates[1])

            # Plot base station locations
            for location in bs_locations:
                x, y, z = location
                ax.scatter(x, y, color='red', marker='^')

            # Define function to update the plot for each time step
            def update(frame):
                ax.clear()
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_xlim(0,self.max_coordinates[0])
                ax.set_ylim(0,self.max_coordinates[1])

                # Plot user locations
                for locations in user_locations:
                    x, y, z = locations[frame]
                    ax.scatter(x, y, color='blue')

                # Plot base station locations
                for location in bs_locations:
                    x, y, z = location
                    ax.scatter(x, y, color='red', marker='^')

                ax.set_title('Time: {}'.format(time_slot[frame]))

            # Create animation object and save the animation as a GIF
            anim = FuncAnimation(fig, update, frames=len(info_list), interval=500)
            #anim.save('locations.gif', writer='pillow', fps=2)

            plt.show()

def render_animation(info_list,max_coordinates):
    fig, axs = plt.subplots()
    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.set_xlim(0, max_coordinates[0])
    axs.set_ylim(0, max_coordinates[1])
    basestations = axs.plot([], [], 'ob')[0]
    users = axs.plot([], [], 'or')[0]

    bs_locations = info_list[0]['bs_locations']

    num_users = len(info_list[0]['user_locations'])
    num_bs = len(bs_locations)
    connections = [axs.plot([], [], '-k')[0] for _ in range(num_users*num_bs)]
                   #for k in itertools.product(range(num_users), range(num_bs))}

    def init():
        basestations.set_data(bs_locations[:, 0], bs_locations[:, 1])
        return basestations, 

    def update(timestep: dict):
        user_locations = timestep['user_locations']
        users.set_data(user_locations[:, 0], user_locations[:, 1])

        for ((user, base_station), connection) in zip(itertools.product(range(num_users), range(num_bs)), connections):
            power_level = timestep['action'][user, base_station]
            linewidth = np.maximum(power_level, 0) * 4
            connection.set_data([user_locations[user, 0], bs_locations[base_station, 0]],
                                [user_locations[user, 1], bs_locations[base_station, 1]])
            connection.set_lw(linewidth)
        return users, basestations, *connections
    
    anim = FuncAnimation(fig, update, frames=info_list, init_func=init,
                         blit=True)
    anim.save("test_data_closest.mp4")