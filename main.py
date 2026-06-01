import osmnx as ox
import numpy as np
import folium
import random
from collections import Counter

# --- 1) Constants ---
# a) Inputs
address = input("Enter starting address (Ex: 450 Jane Stanford Way, Stanford, CA): ")
target_mi = float(input("Enter target running miles: "))
target_m = target_mi * 1609.34

# b) ACO Parameters
ITER = 15 # rounds
ANTS = 25 # ants per round
ALPHA = 3 # pheromone exponent (promotes exploration)
BETA = 4 # heuristic exponent (promotes distance/penalty accuracy)
EVAP = 0.1 # pheromone evap per iteration
REWARD = 400 # pheromone deposit amount
TOP_ANTS = 0.2 # fraction of ants that deposit pheromones
INIT_PHERO = 2 #starting pheromone value

# c) Penalty Costs
REVISIT = 200 # per repeat visit to a node
STOP = 15 # stop sign
DEAD_END = 1000 # node with only one exit
NO_EXIT = 1500 # node with no forward exits
REPEAT_COST = 500 # repeated edge in final circuit score
BACKTRACK = 4 # multiplied by edge length for outbound edges on return leg
TRAVERSED = 300 # flat penalty for reusing an edge already in this trip

# d)Graph and Quadrant Setup
QUADS = {'SW': (-1,-1), 'NW': (-1,1), 'NE': (1,1), 'SE': (1,-1)}
N_TARGETS = 4 # best route tries per quadrant
HIGHWAYS = ('motorway', 'trunk', 'primary') #remove these labeled roadways

# --- 2) Build and Clean Up Map with OSMnx ---
def build_map(lat, lon):
    #Downloads OSMnx map local area and trims to correct radius around start
    print("Downloading map, this may take a few seconds...")
    Map = ox.graph_from_address(address, dist=target_m, dist_type='bbox', network_type='walk', simplify=True)
    Map = ox.project_graph(Map, to_crs="EPSG:4326")  # project to lat/lon coords

    #Remove unwanted road types
    bad_edges = []
    for u, v, k, d in Map.edges(keys=True, data=True):
        road_type = str(d.get('tag', ''))  # gets highway tag string
        if any(t in road_type for t in HIGHWAYS):  # check if any of the map edges are listed dangerous roads
            bad_edges.append((u, v, k))
    Map.remove_edges_from(bad_edges) #remove found dangerous roadways

    # Find closest node to home - for dead end removal below
    start = min(Map.nodes, key=lambda n: (Map.nodes[n]['x']-lon)**2 + (Map.nodes[n]['y']-lat)**2)
    # Remove dead ends (degree <=1)
    while True:
        dead = []
        for n in list(Map.nodes):
            if n != start and Map.degree(n) <= 1: #check not accidentally deleting start node and is dead end
                dead.append(n)
        if not dead:
            break
        Map.remove_nodes_from(dead)#remove dead ends

    # Build OSMnx map dictionaries
    nodes = {}
    for n, d in Map.nodes(data=True): #n - node, d - label
        nodes[n] = d

    edges = {} #ensures every node has an edge list entry
    for n in Map.nodes:
        edges[n] = []

    phero = {}

    # Iterates over every street segment and appends u - start node, v - end node, d - attributes
    for u, v, d in Map.edges(data=True):
        if 'geometry' in d:
            lon, lat = d['geometry'].xy #split long and lat into two arrays
            coord_pairs = zip(lon, lat) #pair current and next nodes together
            geom = [(lat, lon) for lon, lat in coord_pairs] #swap lat and lon order for Folium
        else:
            geom = None

        length = d.get('length') #get street length in meters
        bearing = d.get('bearing', 0.0) #get angle of each edge
        edges[u].append({'nb': v, 'len': length, 'geom': geom, 'bearing': bearing})

        #Same as above but for opposite direction
        if geom:
            reverse_geom = geom[::-1]
        else:
            reverse_geom = None

        reverse_bearing = (bearing + 180) % 360
        edges[v].append({'nb': u,'len': length,'geom': reverse_geom,'bearing': reverse_bearing})
        phero[(u, v)] = phero[(v, u)] = INIT_PHERO

    return nodes, edges, phero, start

# --- 3) Calc Penalties, Lengths, Probabilities for Each Candidate Edge ---
def navigate(nodes, edges, phero, origin, destination, forbidden=None):
    forbidden = forbidden or set() #for return path - inbound edges used
    path = [origin] #nodes visited
    traversed = set() # edges already used this path
    curr = origin #current position (at start is origin)
    for _ in range(450):
        if curr == destination: break
        cands = edges.get(curr, []) #get next edge candidates
        no_backtrack = []
        for e in cands:
            if (len(path) < 2) or not (e['nb'] == path[-2]): #checks if just started or if next edge leads back to where we just came from
                no_backtrack.append(e)
        if no_backtrack:
            cands = no_backtrack #if no options, keep original edges

        #distance between curr and target
        d_curr = np.sqrt((nodes[curr]['x']-nodes[destination]['x'])**2 + (nodes[curr]['y']-nodes[destination]['y'])**2)
        probs = []

        #Penalties calc'd for each edge
        for e in cands:
            nb, length = e['nb'], e['len']
            cost = length
            visits = path.count(nb)
            if visits: #looping back or revisiting node penalty
                cost += REVISIT * visits**2
            if nodes[nb].get('highway') == 'stop': #stop sign penalty
                cost += STOP
            if (curr, nb) in forbidden or (nb, curr) in forbidden: #backtracking penalty (inbound)
                cost += length * BACKTRACK
            if tuple(sorted((curr, nb))) in traversed: #looping back or revisiting edge penalty
                cost += TRAVERSED
            exits = edges.get(nb, []) #dead end penalty
            if len(exits) <= 1:
                cost += DEAD_END
            forward_exits = [] #no exits penalty
            for x in exits:
                if x['nb'] != curr:
                    forward_exits.append(x)
            if not forward_exits:
                cost += NO_EXIT
            if len(path) >= 2: #turn penalty
                for pe in edges.get(path[-2], []):
                    if pe['nb'] == curr:
                        delta = abs(e['bearing'] - pe['bearing']) #calc ange between edges
                        if delta > 180:
                            delta = 360 - delta
                        if delta > 45:
                            cost += 15
                        break

            # Check if neighbor brings you closer to turn target
            d_nb = np.sqrt((nodes[nb]['x']-nodes[destination]['x'])**2 + (nodes[nb]['y']-nodes[destination]['y'])**2)
            if d_curr > d_nb:
                pull = 3.0 + (d_curr - d_nb)/(length + 1e-6) #higher pull used to bias eta for the edge
            else:
                pull = 0.05 #lower pull used to bias eta against the edge

            tau = phero.get((curr, nb), INIT_PHERO) #pheromone levels
            eta = (1.0 / cost) * pull #edge weights (length + penalties + pull)
            probs.append((tau**ALPHA) * (eta**BETA)) #Eqn 22.15

        #Calc total probability - Eqn 22.16
        total = sum(probs)
        if total > 0:
            norm_weights = [p/total for p in probs] #normalize all probs
            chosen = random.choices(cands, weights=norm_weights, k=1)[0] #choose edge,higher prob has higher chance of being chosen
        else:
            chosen = random.choice(cands)

        path.append(chosen['nb']) #append chosen neighbor
        traversed.add(tuple(sorted((path[-2], path[-1])))) #add edge just chosen to traversed for future penalties
        curr = chosen['nb'] #advance ant position
    return path

# --- 4) ACO Algorithm ---
def find_target(nodes, home, sx, sy, radius):
    #Nodes closest to the ideal turnaround radius in the given quadrant
    candidates = []
    for n, d in nodes.items():
        dx, dy = d['x'] - home['x'], d['y'] - home['y']
        if np.sign(dx) != sx or np.sign(dy) != sy:
            continue
        dist_from_radius = abs(np.sqrt(dx**2 + dy**2) * 111000 - radius)
        candidates.append((dist_from_radius, n))
    return [n for _, n in sorted(candidates)[:N_TARGETS]]

def score_circuit(edges, circuit):
    # Sum the length of each edge in the circuit
    dist = 0
    for u, v in zip(circuit[:-1], circuit[1:]):
        for e in edges.get(u, []):
            if e['nb'] == v:
                dist += e['len']
                break

    # Collect all edge pairs as undirected tuples
    pairs = []
    for u, v in zip(circuit[:-1], circuit[1:]):
        pair = tuple(sorted((u, v)))
        pairs.append(pair)

    # Penalize edges that appear more than once
    repeat = 0
    for pair, count in Counter(pairs).items():
        if count > 1:
            repeat += REPEAT_COST * (count - 1)

    return abs(dist - target_m) + repeat, dist

def deposit_pheromones(phero, solutions):
    #Top 20% ants deposit pheromones on their routes
    top = solutions[:max(1, int(len(solutions) * TOP_ANTS))]
    for circuit, score, _ in top:
        for u, v in zip(circuit[:-1], circuit[1:]):
            if (u, v) in phero:
                phero[(u, v)] += REWARD / (1 + score)
                phero[(v, u)] += REWARD / (1 + score)

def run_aco(nodes, edges, phero, start):
    radius = (target_m / (2 * np.sqrt(2))) * 0.85 #ideal turnaround dist
    best_path, best_score, best_dist = None, float('inf'), 0 #initialize
    home = nodes[start] #start node - to meas dist

    for quad, (sx, sy) in QUADS.items():
        for milestone in find_target(nodes, home, sx, sy, radius): #find target for each quad
            for k in phero: phero[k] = INIT_PHERO #make sure all pheromones reset before new try
            q_best = float('inf') #track best score

            for i in range(ITER): #runs through designated iterations
                solutions = []
                for _ in range(ANTS): #releases designated ants
                    # Outbound: home -> milestone
                    out = navigate(nodes, edges, phero, start, milestone) #ant navigates to target
                    if not out or out[-1] != milestone: continue #if ant failed, skip

                    # Inbound: milestone -> home, avoiding outbound edges
                    out_edges = set(zip(out[:-1], out[1:])) #pair consecutive nodes
                    back = navigate(nodes, edges, phero, milestone, start, out_edges) # ant navigates back to home
                    if not back or back[-1] != start: continue #if ant failed, skip

                    circuit = out + back[1:] #drops last home node of path
                    score, dist = score_circuit(edges, circuit) #route scored
                    solutions.append((circuit, score, dist))
                    if score < best_score: #check if that is best score
                        best_score, best_path, best_dist = score, circuit, dist

                for k in phero: phero[k] *= (1 - EVAP) #apply provided evaporation after all ants finish
                if solutions: #sort and take best route scored
                    solutions.sort(key=lambda x: x[1])
                    deposit_pheromones(phero, solutions)
                    q_best = min(q_best, solutions[0][1])
        print(f"  Quadrant: {quad}, Best Score: {q_best:.1f}")

    return best_path, best_dist

# --- 6) Save Folium Map ---
def save_map(nodes, edges, route, lat, lon):
    coords = []
    for u, v in zip(route[:-1], route[1:]):
        # Find the edge between u and v
        matching_edge = None
        for e in edges.get(u, []):
            if e['nb'] == v:
                matching_edge = e
                break

        # Use edge geometry if available, otherwise draw a straight line between nodes
        if matching_edge and matching_edge['geom']:
            pts = matching_edge['geom']
        else:
            node_u = (nodes[u]['y'], nodes[u]['x'])
            node_v = (nodes[v]['y'], nodes[v]['x'])
            pts = [node_u, node_v]

        for pt in pts:
            if not coords or coords[-1] != pt: coords.append(pt)

    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.PolyLine(coords, color='blue', weight=6).add_to(m)
    folium.Marker([lat, lon], popup="Home", icon=folium.Icon(color='blue', icon='home')).add_to(m)
    m.save("aco_route.html")

# --- 7) Main - Run Everything ---
def main():
    lat, lon = ox.geocode(address)
    nodes, edges, phero, start = build_map(lat, lon)
    route, distance = run_aco(nodes, edges, phero, start)
    save_map(nodes, edges, route, lat, lon)
    print(f"\nDone! {distance/1609.34:.2f} miles | Open aco_route.html to view.")

if __name__ == "__main__":
    main()