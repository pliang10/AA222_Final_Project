# AA222_Final_Project
## Generates a loop running route from a designated starting point using Ant Colony Optimization.

### 1. Requirements

Ensure you have the following Python libraries installed:

<pre>
pip install osmnx numpy folium
</pre>

### 2. Run main.py

The file 'main.py' has the project code that will generate the running route using Ant Colony Optimization (ACO). Run the code and enter your address and target running mileage as prompted. A html file with the final running route visualized will be saved under aco_route.html.

### How It Works
The algorithm splits each route attempt into two legs:

* Outbound — an ant navigates from home to a turnaround milestone node located roughly halfway along the target distance
* Inbound — a second ant navigates back home, penalized for reusing outbound streets,
    
This is repeated across 25 ants, 15 iterations, and 4 quadrants. The circuit with the lowest score (distance error + penalties) is saved as the final route. Edge selection at each step follows the standard ACO probability formula - weighted by pheromone strength and a heuristic that combines inverse edge cost and directional pull toward the milestone.
