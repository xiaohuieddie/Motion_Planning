
## Project: 3D Motion Planning
![Quad Image](./misc/enroute.png)

---


# Required Steps for a Passing Submission:
1. Load the 2.5D map in the colliders.csv file describing the environment.
2. Discretize the environment into a grid or graph representation.
3. Define the start and goal locations.
4. Perform a search using A* or other search algorithm.
5. Use a collinearity test or ray tracing method (like Bresenham) to remove unnecessary waypoints.
6. Return waypoints in local ECEF coordinates (format for `self.all_waypoints` is [N, E, altitude, heading], where the drone’s start location corresponds to [0, 0, 0, 0].
7. Write it up.
8. Congratulations!  Your Done!

## [Rubric](https://review.udacity.com/#!/rubrics/1534/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it! Below I describe how I addressed each rubric point and where in my code each point is handled.

### Explain the Starter Code

#### 1. Explain the functionality of what's provided in `motion_planning.py` and `planning_utils.py`
These scripts contain a basic planning implementation that includes the following contents:
* The main code to command the drone like the 'backyardflyer' project
* Event programming: There will be a transition between different flying states like 'TakeOff', 'Arming', 'Landing', etc. In this strategy, the drone can be responsive accordingly even though there is an obstacle while it is flying.
* Grid creation: Based on the given data, the function 'crate_grid' created a grid space, in which the obstacle is marked as `1`
* Searching Algorithm: A* searching algorithm is used to search the closest path to the targeted position

### Implementing Your Path Planning Algorithm

#### 1. Set your global home position
I use `np.loadtxt` to read the data of `colliders.csv` and make the data type to be String. `[0]` is used to only retrive the first row data which is the global home position. In order to get the numeric value, I used the function `split()` to split the string like `lat0 37.792480` by space and then only read the numeric value. Finally, I convert the data type from 'String' to 'Float' through `float()`.

``` python
    # TODO: read lat0, lon0 from colliders into floating point values
    lat0, lon0= np.loadtxt('colliders.csv', delimiter=',', dtype='str', usecols = (0,1))[0]
    lat0 = float(lat0.split()[1])
    lon0 = float(lon0.split()[1])

    # TODO: set home position to (lon0, lat0, 0)
    self.set_home_position(lon0, lat0, 0)
    
```

#### 2. Set your current local position
I set the local position relative to the global home position using the following line:

``` python
        global_home = self.global_home

        # TODO: retrieve current global position
        global_position = self.global_position
 
        # TODO: convert to current local position using global_to_local()
        local_position = global_to_local(global_position, global_home)
```
I used the function `global_to_local` with two parameters `global_position` and `global_home` to convert the global position to local position, which is relative position with respect to home position.

#### 3. Set grid start position from local position
This is another step in adding flexibility to the start location. As long as it works you're good to go!
``` python
start = (int(current_local_pos[0]+north_offset),
int(current_local_pos[1]+east_offset))
```

I have taken into account the north and east offset on the map to find the place in the grid.
#### 4. Set grid goal position from geodetic coords
The goal is set using the two lines:

```python
        # TODO: adapt to set goal as latitude / longitude position and convert
        goal_global_position = (-122.39827005, 37.79639587, 0)
        goal_local_position = global_to_local(goal_global_position, global_home)
        grid_goal = (int(goal_local_position - north_offset),int(goal_local_position[1] - east_offset))
        print('Local Start and Goal: ', grid_start, grid_goal)
```

As you can see the input is in geodetic coordinates `(-122.39827005, 37.79639587, 0)` from which I retrieve the local coordinates using `global_to_local`. The user can also select their goal from running the script with arguments s:


```
python motion_planning.py --lat 37.79639587 --lon
-122.39827005
```

This was done by adding two arguments:
```
parser.add_argument('--lat', type=float, default=1000, help="latitude")
```
```
parser.add_argument('--lon', type=float, default=1000, help="latitude")
```

#### 5. Modify A* to include diagonal motion (or replace A* altogether)
Instead of using the grid search, I modified the A* to be adaptive to graph space search.
Firstly I modifed the function `create_grid` with adding following code:
```python
    # create a voronoi graph based on the location of obstacle centres
    graph = Voronoi(points)
```
The intension of the code above is try to extract the voronoi graph based on the obstacle center points. Then I also checked each edge of the gragh to see if it is in free space.

Secondly, I added weight for graph as following:
``` python
    # Add weight to graph
    G = nx.Graph()
    for e in edges:
        p1 = tuple(e[0])
        p2 = tuple(e[1])
        dist = LA.norm(np.array(p2) - np.array(p1))
        G.add_edge(p1, p2, weight=dist)
```

Then, the final step before starting our A* search is trying to find the closest start & goal points in the graph using `closest_point`

Finally, I modified the A* initial code for grid space to be:
```python
        for next_node in graph[current_node]:
            cost = graph.edges[current_node, next_node]['weight']
            branch_cost = current_cost + cost
            queue_cost = branch_cost + h(next_node, goal)

        if next_node not in visited:                
            visited.add(next_node)               
            branch[next_node] = (branch_cost, current_node)
            queue.put((queue_cost, next_node))
```

#### 6. Cull waypoints 
In collinearity I select continuous groups of points (3 in each step) to see if they belong in a line or approximately belong to a line. If they can be connected to a line I replace the two waypoints with a single one (longer) and continue the search to see if I can add more way points to the same line. 
```python
def point(p):
    return np.array([p[0], p[1], 1.]).reshape(1, -1)

def prune_path(path):
    pruned_path = [p for p in path]
    
    i = 0
    while i < len(pruned_path) - 2:
        p1 = point(pruned_path[i])
        p2 = point(pruned_path[i+1])
        p3 = point(pruned_path[i+2])
        
        # If the 3 points are in a line remove
        # the 2nd point.
        # The 3rd point now becomes and 2nd point
        # and the check is redone with a new third point
        # on the next iteration.
        if collinearity_check(p1, p2, p3):
            # Something subtle here but we can mutate
            # `pruned_path` freely because the length
            # of the list is check on every iteration.
            pruned_path.remove(pruned_path[i+1])
        else:
            i += 1
    return pruned_path
```


### Execute the flight
#### 1. Does it work?
It works!

### Double check that you've met specifications for each of the [rubric](https://review.udacity.com/#!/rubrics/1534/view) points.
  
# Extra Challenges: Real World Planning

For an extra challenge, consider implementing some of the techniques described in the "Real World Planning" lesson. You could try implementing a vehicle model to take dynamic constraints into account, or implement a replanning method to invoke if you get off course or encounter unexpected obstacles.




