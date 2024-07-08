"""
    A Python program to find Pareto optimal solutions. The complexity of the simple cull 
    algorithm is O(n^2). The complexity of the check_nondomination is also O(n^2).
    The program requires numpy and matplotlib to run.
    Packages required: numpy, matplotlib, rich, pymoo
"""
import numpy as np
from math import sqrt
from pymoo.factory import get_performance_indicator

def simple_cull(inputPoints, dominates):
    """
    Simple cull algorithm
    Ref: Geilen, M., & Basten, T. (2007, April). A calculator for Pareto points. In 2007 Design, Automation & Test in Europe Conference & Exhibition (pp. 1-6). IEEE.
    https://www.es.ele.tue.nl/pareto/papers/date2007_paretocalculator_final.pdf
    """
    paretoPoints = set() # Set of pareto points
    candidateRowNr = 0 # A constant number to take first row always
    dominatedPoints = set() # Set of dominated points
    while True:
        candidateRow = inputPoints[candidateRowNr] # Take the first row from inputPoints
        inputPoints.remove(candidateRow) # remove candidateRow from inputPoints, so that we won't deal with it again
        rowNr = 0
        nonDominated = True # Assuming the candidateRow is non-dominated
        while len(inputPoints) != 0 and rowNr < len(inputPoints): # Loop while there are points in inputPoints and points that are not processed
            row = inputPoints[rowNr] # Take the row
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                # If the candidateRow gets dominated
                nonDominated = False # Assumption was wrong, flag is as dominated
                dominatedPoints.add(tuple(candidateRow)) # Add candidate to dominatedPoints
                rowNr += 1 # Increase row number by 1
            else:
                rowNr += 1 # Increase row number by 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0: #Break from loop if there are no points left
            break
    return paretoPoints, dominatedPoints

def weakly_dominates(row, candidateRow):
    """
        Domination requires all of row's attributes are less than or equal to 
        candidateRow's attributes. A shortcut applied here using list comprehensions
        returns true if all of the conditions are met.
    """
    #return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)  
    return all([row[x] <= candidateRow[x] for x in range(len(row))])

def strongly_dominates(p1, p2):
    """
        Returns true if p1 dominates p2. Strong domination requires all of p1's attributes are less 
        than or equal to p2's attributes, at least one of them is less than p2's attributes.
    """
    return all([p1[x] <= p2[x] for x in range(len(p1))]) and any([p1[x] < p2[x] for x in range(len(p1))])

def read(file):
    """
        Reads the input txt file and returns the points as list of tuples
    """
    points = [] # Create an empty list
    with open(file) as f: # Open the file
        lines = f.readlines() # Read all lines
    for line in lines[1:]: # For every line except the header line
        points.append(tuple(int(i) for i in line.split(" "))) # Split line and convert them to integer
    headers =lines[0].split(" ")
    return points, headers

def draw(paretoPoints, dominatedPoints, headers, plot_title=None):
    """
        Draws the pareto and dominated sets
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion() # Added to draw multiple plots at once. plt.show() blocks the execution.
    fig = plt.figure() # Create figure
    if plot_title: # If plot_title is not None, if there is a parameter in plot_title
        plt.title(plot_title) # Change plot title
    ax = fig.add_subplot(111, projection='3d') # Create a 3D plot area
    ax.set_xlabel(headers[0]) # Change labels
    ax.set_ylabel(headers[1]) # Change labels
    ax.set_zlabel(headers[2]) # Change labels

    dp = np.array(list(dominatedPoints)) # Convert to numpy arrays for slicing
    pp = np.array(list(paretoPoints)) # Convert to numpy arrays for slicing
    print(f"Pareto set: {pp.shape},\nDominated set:{dp.shape}")
    ax.scatter(dp[:,0],dp[:,1],dp[:,2]) # Draw dominated points
    ax.scatter(pp[:,0],pp[:,1],pp[:,2],color='red')# Draw non-dominated points

    if len(pp)>1:
        import matplotlib.tri as mtri
        triang = mtri.Triangulation(pp[:,0],pp[:,1]) 
        ax.plot_trisurf(triang,pp[:,2],color='green', alpha=0.3) # Draw pareto front surface as triangular shapes
    plt.show() # Show figure
    plt.pause(0.0001) # Added to draw multiple plots at once. plt.show() blocks the execution.

def check_nondomination(pareto, dominated, dominates):
    for p in pareto: # For every points in pareto
        for d in dominated: # For every points in dominated
            if dominates(d, p): # If any point in dominated set dominates any point in pareto set, return false
                return False
    return True # If none of the pareto set is dominated, return true

def pointdistance(p1,p2):
    return sqrt(sum([(xi-yi)**2 for xi,yi in zip(p1,p2)])) # Return euclidian distance between 2 points


def distance(set1, set2): # They are actually lists to be able to subcscript
    gd = 0 # Variable to store general distance
    r = len(set1) # Number of values in set 1(Corrected)
    for x in set1: # For every point, x, in set 1
        d =  pointdistance(x, set2[0])# Set distance as the distance of the first element from set 2
        for y in set2[1:]: # For every points in set 2 except the first one, we already used it to calculate distance above
            dcandidate = pointdistance(x,y) # candidate distance, if it is smaller than already seen minimum distance change it
            if dcandidate < d:
                d = dcandidate
        gd += d*d # Add square of the distance as in the equation 4 of paper "A Convergence indicator for Multi-Objective Optimization Algorithms"
    return sqrt(gd)/r # Calculate the distance

def get_reference_point(list_of_point_lists):
    """
        Returns the reference point of the list of point lists. Reference point is the point with the
        biggest values in each dimension.
    """
    reference_point = list_of_point_lists[0][0] # Take the first point from the first list
    for point_list in list_of_point_lists: # For every point list in the list of point lists
        for point in point_list: # For every point in the point list
            reference_point = [max(reference_point[i], point[i]) for i in range(len(reference_point))] # Take the maximum of the reference point and the point
    return reference_point

def average_hausdorff_distance(list1, list2):
    """
        Calculates the average hausdorff distance between two sets of points
    """
    gd = distance(list1, list2)
    igd = distance(list2, list1)
    return max(gd, igd)

def calculate_distances_and_print(point_group, pareto, file_names):
    from rich.console import Console # Rich library for printing
    from rich.table import Table # Rich library for printing
    from math import isclose # Library for checking if two numbers are close to each other
    console = Console() # Create console
    table = Table(title="Distances", header_style="bold magenta") # Create table
    table.add_column("Measurement", justify="left")
    # Add column for each file name
    for file_name in file_names:
        table.add_column(file_name, justify="right")

    gd2 = [distance(points, pareto) for points in point_group] # Calculate the distance between the pareto_final and every point in the point group
    igd2 = [distance(pareto, points) for points in point_group] # Calculate the distance between the pareto_final and every point in the point group
    table.add_row("GD", *[f"{'[green]'+f'{d:.2f}'+'[/green]' if isclose(d, min(gd2)) else f'{d:.2f}'}" for d in gd2]) # Add the distance to the table
    table.add_row("IGD", *[f"{'[green]'+f'{d:.2f}'+'[/green]' if isclose(d, min(igd2)) else f'{d:.2f}'}" for d in igd2]) # Add the distance to the table
    gdf = get_performance_indicator("gd", pareto) # Calculate the performance indicator using pareto_final
    gd  = [gdf.do(points) for points in point_group] # Calculate the performance indicator for every point in the point group
    igdf = get_performance_indicator("igd", pareto) # Calculate the performance indicator using pareto_final
    igd  = [igdf.do(points) for points in point_group] # Calculate the performance indicator for every point in the point group
    table.add_row("GD(p=1)", *[f"{'[green]'+f'{d:.2f}'+'[/green]' if isclose(d, min(gd)) else f'{d:.2f}'}" for d in gd]) # Add the distance to the table
    table.add_row("IGD(p=1)", *[f"{'[green]'+f'{d:.2f}'+'[/green]' if isclose(d, min(igd)) else f'{d:.2f}'}" for d in igd]) # Add the distance to the table
    gdplusf = get_performance_indicator("gd+", pareto) # Calculate the performance indicator using pareto_final
    gdplus  = [gdplusf.do(points) for points in point_group] # Calculate the performance indicator for every point in the point group
    igdplusf = get_performance_indicator("igd+", pareto) # Calculate the performance indicator using pareto_final
    igdplus  = [igdplusf.do(points) for points in point_group] # Calculate the performance indicator for every point in the point group
    table.add_row("GD+", *[f"{'[green]'+f'{d:.2f}'+'[/green]' if isclose(d, min(gdplus)) else f'{d:.2f}'}" for d in gdplus]) # Add the distance to the table
    table.add_row("IGD+", *[f"{'[green]'+f'{d:.2f}'+'[/green]' if isclose(d, min(igdplus)) else f'{d:.2f}'}" for d in igdplus]) # Add the distance to the table
    hausdorff2 = [average_hausdorff_distance(points, pareto) for points in point_group] # Calculate the average hausdorff distance between the pareto_final and every point in the point group
    table.add_row("Hausdorff", *[f"{'[green]'+f'{d:.2f}'+'[/green]' if isclose(d, min(hausdorff2)) else f'{d:.2f}'}" for d in hausdorff2]) # Add the distance to the table
    hausdorff = [max(d1,d2) for d1, d2 in zip(gd, igd)] # Calculate the average hausdorff distance between the pareto_final and every point in the point group
    table.add_row("Hausdorff(p=1)", *[f"{'[green]'+f'{d:.2f}'+'[/green]' if isclose(d, min(hausdorff)) else f'{d:.2f}'}" for d in hausdorff]) # Add the distance to the table
    hausdorffplus = [max(d1,d2) for d1, d2 in zip(gdplus, igdplus)] # Calculate the average hausdorff distance between the pareto_final and every point in the point group
    table.add_row("Hausdorff+", *[f"{'[green]'+f'{d:.2f}'+'[/green]' if isclose(d, min(hausdorffplus)) else f'{d:.2f}'}" for d in hausdorffplus]) # Add the distance to the table
    ref_point = get_reference_point(point_group+[pareto]) # Calculate the reference point as the maximum values of the each axis of the pareto_final and every point in the point group
    hypervolumef = get_performance_indicator("hv", ref_point) # Calculate the performance indicator using the reference point
    hypervolumes = [hypervolumef.do(points) for points in point_group] # Calculate the performance indicator for every point in the point group
    table.add_row("Hypervolume", *[f"{'[green]'+f'{d:.2f}'+'[/green]' if isclose(d, max(hypervolumes)) else f'{d:.2f}'}" for d in hypervolumes]) # Add the distance to the table
    hypervolume_pareto = hypervolumef.do(pareto) # Calculate the hypervolume of the pareto_final
    table.add_row("Pareto HV", *[f"{hypervolume_pareto:.2f}"]*len(file_names)) # Add the distance to the table

    console.print(table)

def main():
    dominates = strongly_dominates # Set the domination function as the strongly_dominates function
    files = [ #File list, add more items if there are more files.
        "results_arb_b_h1f2.txt",
        "results_bound_b_pp1.txt",
        "results_bound_brs_h2f8.txt",
        "results_xyz.txt"
    ]
    headers = ["latency","energy","cost"]
    pareto_list_of_sets = [simple_cull(read(file)[0], dominates) for file in files] # Read all files and calculate pareto fronts of each
    for i in range(len(pareto_list_of_sets)): # loop over pareto_list_of_sets by using index variable i
        pareto, dominated = pareto_list_of_sets[i] # get pareto adn dominated into distinct variables
        draw(pareto, dominated, headers, files[i]) #  draw pareto fronts of 4 files
    paretoset = set() # Variable to store all pareto points from files
    dominated = set() # Variable to store all dominated points from files
    for s,d in pareto_list_of_sets: # Combine all pareto and dominated points into 2 sets
        paretoset = paretoset.union(s)
        dominated = dominated.union(d)

    print(paretoset)
    print(dominated)
    draw(paretoset, dominated, headers, "Final pareto") # Plot all pareto points and dominated points
    pareto_final, _ = simple_cull(list(paretoset), dominates) #Calculate pareto final from pareto points set.Convert to list to use subscripts
    print(pareto_final) # Print points in pareto final
    distances = [distance(list(pareto_final), list(i[0])) for i in pareto_list_of_sets] # Calculate distances of all pareto fronts to pareto final
    print(f"Distances: {distances}") # Print distances
    print("Distances ordered:")
    for file, dist in sorted(zip(files, distances),key=lambda item:item[1]): # Print file names ordered by distances ascending
        print(f"File: {file} Distance: {dist}")
    calculate_distances_and_print([np.array(list(p[0])) for p in pareto_list_of_sets], np.array(list(pareto_final)), files) # Calculate distances and print them
    input("Press Enter to exit") # When plt.show() is nt blocked anymore, program just ends and plots are gone
if __name__ == "__main__":
    main()