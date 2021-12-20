"""
Functions to calculate pemdiv and emdiv

Author: MPC

Version 0.1.1
"""

from ortools.graph import pywrapgraph
import numpy as np


class PemdivError(Exception):
    "base exception for pemdiv module"

    def __init__(self, message=None):
        if message is None:
            message = "A non-descript PemdivError occured"
        super(PemdivError, self).__init__(message)


class NonIntegerDetectedError(PemdivError):
    "Non-int found, when int was needed"

    def __init__(self):
        super(NonIntegerDetectedError, self).__init__(
            message="Non-int found, when int was needed"
        )


class SupplyDemandMismatchError(PemdivError):
    "sum of supplies/demands across nodes was not zero, when expected"

    def __init__(self):
        super(SupplyDemandMismatchError, self).__init__(
            message="sum of supplies/demands was nonzero when expected"
        )


class EdgesLenMismatchError(PemdivError):
    "Number of implied edges in lists across the edges dictionary do not match"

    def __init__(self):
        super(EdgesLenMismatchError, self).__init__(
            message="# of edges does not match across tails, heads, capacities, etc"
        )


class NodeSuppliesMismatchError(PemdivError):
    "Number of nodes does not equal items in the supplies obj provided"

    def __init__(self):
        super(NodeSuppliesMismatchError, self).__init__(
            message="# of nodes does not = items in supplies obj"
        )


class NodeIdError(PemdivError):
    "Node ids are not ints that follow sequence range(0,num_nodes)"

    def __init__(self):
        super(NodeIdError, self).__init__(
            message="Node ids do not follow pattern range(0, num_nodes)"
        )


def check_sum_to_0(supply_dict):
    "check supply_dict sums to 0 (normalized supply/demand check)"
    if np.sum([val for _, val in supply_dict.items()]) == 0:
        pass
    else:
        raise SupplyDemandMismatchError


def is_int(obj):
    "check if an object is an int"
    if isinstance(obj, int):
        pass
    else:
        raise NonIntegerDetectedError


def check_int_inputs(edge_dict, node_supplies_dict, extra_cost_per_unit):
    """
    Check that all inputs to calc_pemdiv or calc_emdiv are as expected
     edge_dict -- dictionary with lists of int items (keys are not ints)
     node_supplies_dict -- dictionary where keys are ints and values are ints
     extra_cost_per_unit -- int
    """
    # extra_cost_per_unit should be int
    try:
        is_int(extra_cost_per_unit)
    except NonIntegerDetectedError as err:
        print(err + "; problem in {}".format("extra_cost_per_unit"))
    # edge_dict should be dict with only ints in the values (keys are not ints)
    for __, val in edge_dict.items():
        if isinstance(val, list):
            for item in val:
                try:
                    is_int(item)
                except NonIntegerDetectedError as err:
                    print(err + "; problem in edge_dict")
        else:
            print("Problem in edge_dict, dict keys to lists expected")
    # node_supplies_dict should be a dict with keys of ints and supply values ints
    for key, val in node_supplies_dict.items():
        try:
            is_int(key)
        except NonIntegerDetectedError as err:
            print(err + "; problem in node_supplies_dict keys")
        for item in val:
            try:
                is_int(key)
            except NonIntegerDetectedError as err:
                print(err + "; problem in node_supplies_dict values")


def are_equal_len(dict_of_lists):
    "return true if all lists in a dict are of equal length"
    it_dol = iter(dict_of_lists)
    n_first = len(dict_of_lists[next(it_dol)])
    if not all(len(dict_of_lists[key]) == n_first for key in it_dol):
        raise EdgesLenMismatchError


def check_implied_nodes_num(supply_list, node_id_list):
    """
    check if all the lists that define nodes imply the same
    number of nodes
    """
    try:
        # are equal needs a dict
        are_equal_len({0: supply_list, 1: node_id_list})
    except NodeSuppliesMismatchError as err:
        print(err)


def check_implied_edges_num(dict_of_lists):
    """
    check if all the lists (in dict) that define the edges imply the same
    number of edges
    """
    try:
        are_equal_len(dict_of_lists)
    except EdgesLenMismatchError as err:
        print(err)


def count_nodes_from_edges(edge_heads, edge_tails):
    "count number of unique head and tail nodes"
    num_nodes = len(set(list(set(edge_heads)) + list(set(edge_tails))))
    return num_nodes


def is_seq_0_to_num(set_of_ids):
    "Are a set of (int) ids equal to seq range(0, num_nodes)"
    num_nodes = len(set_of_ids)
    if set_of_ids != set(range(0, num_nodes)):
        raise NodeIdError


def check_edge_node_ids(edge_heads, edge_tails):
    "check that ids are a proper int sequence"
    try:
        is_seq_0_to_num(set(edge_heads + edge_tails))
    except NodeIdError as err:
        print(err)


def check_supplies_node_id(node_supplies_dict):
    """
    Check node ids are ints as keys in node_supplies_dict

    """
    key_ids = {int(key) for key in node_supplies_dict}
    try:
        is_seq_0_to_num(key_ids)
    except NodeIdError as err:
        print(err)


def is_empty(any_structure):
    "check if an object is empty of not"
    if any_structure:
        return False
    return True


def make_edges_dict(edge_heads, edge_tails, capacities, unit_costs):
    """
    Make a dictionary of edges
    Inputs
     edge_heads: A list of all node heads for edges, integers starting at 0, ints
     edge_tails: A list of all node tails for edges, integers starting at 0, ints
     capacities: A list of capacities for edges, must be integers, ints
     unit_costs: A list of costs to transport 1 unit across edge, must be integers, ints
    Output
     A dictionary of lists, also prints number of edges and nodes implied
     Note: checks for whether implied nodes are equal across lists
    """
    check_edge_node_ids(edge_heads, edge_tails)
    dict_of_lists = {
        "edge_heads": edge_heads,
        "edge_tails": edge_tails,
        "capacities": capacities,
        "unit_costs": unit_costs,
    }
    check_implied_edges_num(dict_of_lists=dict_of_lists)
    print("Num of implied edges: {}".format(len(edge_heads)))
    num_nodes = count_nodes_from_edges(edge_heads, edge_tails)
    print("Num of implied nodes: {}".format(num_nodes))
    return dict_of_lists


def make_nodes_supplies_dict(node_supplies_list, node_ids=None):
    """
    make a dictionary with node id as key
    and supply (can be pos or neg) as value
    Input:
        node_supplies_list -- a list of supplies in border
        node_ids -- int sequential ids, optional (in same order as node_supplies_list)
    Note: node ids need to be ints and so are assumed to be the slots
    in the list for each value. This is checked in pemdiv.calc_pemdiv
    Example [1,-10,-1] --> {"0": 1, "1": -10, "2": -1}
    """
    if is_empty(node_ids):
        print(
            "No node_ids provided, assuming nodeSuppliesList is in order (0,num_nodes)"
        )
        return {i: val for i, val in enumerate(node_supplies_list)}
    check_implied_nodes_num(node_supplies_list, node_ids)
    zip_it = zip(node_ids, node_supplies_list)
    return {node[0]: node[1] for node in zip_it}


def toy_1d_data(supplies_equal=True):
    "Make toy 1d data for pemdiv example"
    data = {
        "start_nodes": [0, 0, 1, 1, 1, 2, 2, 3, 4],
        "end_nodes": [1, 2, 2, 3, 4, 3, 4, 4, 2],
        "capacities": [
            20000,
            20000,
            20000,
            20000,
            20000,
            20000,
            20000,
            20000,
            20000,
        ],
        "unit_costs": [4, 4, 2, 2, 6, 1, 3, 2, 3],
        "extra_cost_per_unit": 4,
    }
    if supplies_equal is True:
        data["supplies"] = [20, 0, 0, -5, -15]
    else:
        data["supplies"] = [14, 1, 0, -5, -15]
    return data


def calc_pemdiv(
    edge_dict,
    node_supplies_dict,
    extra_cost_per_unit,
    solver="maxFlowWithMinCost",
    print_out=False,
):
    """
    Inputs:
       edge_dict: usually from make_edges_dict, includes key, value pairs
           "edge_heads": A list of all node heads for edges, integers starting at 0, ints
           "edge_tails": A list of all node tails for edges, integers starting at 0, ints
           "capacities": A list of capacities for edges, must be integers, ints
           "unit_costs": A list of costs to transport 1 unit across edge, must be integers, ints
       node_supplies_dict: A dictionary of initial supplies for nodes,
           node int id as key
           supply as value, must be int.
           Note: positive is supply, negative is demand. This should be defined
               such that, for source distribution A, and target distribution B the
               supply distribution is p(A-B), with the subtraction being node by node, ints
        extra_cost_per_unit: a scalar, the cost for destroying or creating one unit, if supply
                    does not equal demand.
     solver: either "minCost" for symmetric edges that are normed, or "maxFlowWithMinCost"
       for anything else (default = maxFlowWithMinCost)
     print_out: print out flow, True or False (default=True)
    Output:
      a print out of results if print_out==True
      a dictionary, with keys:
        score: the pemdiv score, in units of work (lower is better)
        minCostFlow: the minCostFlowInstance object
        massCreatDestroyCost: the total costs of mass creation and destruction
    Note: if sum of nodeSupplies is 0 and edges are symmetric then use
      .Solve, otherwise use .SolveMaxFlowWithMinCost
    """
    # check input
    check_supplies_node_id(node_supplies_dict)
    check_edge_node_ids(edge_dict["edge_heads"], edge_dict["edge_tails"])
    # check_int_inputs()
    # set up whether this is a normed problem (abs_sum_node_supplies==0), and what
    # total supply and total demand are (these last two are used in creation/destruction cost)
    abs_sum_node_supplies = abs(
        np.array([val for key, val in node_supplies_dict.items()]).sum(0)
    )
    total_supply_to_move = np.array(
        [val for key, val in node_supplies_dict.items() if val > 0]
    ).sum(0)
    total_demand = np.array(
        [val for key, val in node_supplies_dict.items() if val < 0]
    ).sum(0)

    minCostFlowInstance = pywrapgraph.SimpleMinCostFlow()
    for i in range(0, len(edge_dict["edge_heads"])):
        minCostFlowInstance.AddArcWithCapacityAndUnitCost(
            edge_dict["edge_heads"][i],
            edge_dict["edge_tails"][i],
            edge_dict["capacities"][i],
            edge_dict["unit_costs"][i],
        )
    for key, val in node_supplies_dict.items():
        minCostFlowInstance.SetNodeSupply(key, val)
    if solver == "minCost":
        if abs_sum_node_supplies != 0:
            print(
                "supply does not equal demand, change to solver='maxFlowWithMinCost'"
            )
        if minCostFlowInstance.Solve() == minCostFlowInstance.OPTIMAL:
            if print_out:
                print("Minimum cost:", minCostFlowInstance.OptimalCost())
                print("")
                print("  Arc    Flow / Capacity  Cost")
                for i in range(minCostFlowInstance.NumArcs()):
                    cost = minCostFlowInstance.Flow(
                        i
                    ) * minCostFlowInstance.UnitCost(i)
                    print(
                        "%1s -> %1s   %3s  / %3s       %3s"
                        % (
                            minCostFlowInstance.Tail(i),
                            minCostFlowInstance.Head(i),
                            minCostFlowInstance.Flow(i),
                            minCostFlowInstance.Capacity(i),
                            cost,
                        )
                    )
        else:
            print("There was an issue with the min cost flow input")
            print(
                "Note: .Solve tried, you might want to switch to solver=MaxFlowWithMinCost"
            )
    else:
        if (
            minCostFlowInstance.SolveMaxFlowWithMinCost()
            == minCostFlowInstance.OPTIMAL
        ):
            mass_to_destroy = abs(
                total_supply_to_move - minCostFlowInstance.MaximumFlow()
            )
            mass_to_create = abs(
                -total_demand - minCostFlowInstance.MaximumFlow()
            )
            create_destroy_cost = (
                mass_to_create + mass_to_destroy
            ) * extra_cost_per_unit
            if print_out:
                print("==========================")
                print(
                    "pEMDiv: %3s"
                    % (minCostFlowInstance.OptimalCost() + create_destroy_cost)
                )
                print("==========================")
                print("Minimum Flow cost:", minCostFlowInstance.OptimalCost())
                print("Extra Deficit/Surplus cost:", create_destroy_cost)
                print("Maximum Flow was: ", minCostFlowInstance.MaximumFlow())
                print("")
                print("  Arc    Flow / Capacity  Cost")
                for i in range(minCostFlowInstance.NumArcs()):
                    cost = minCostFlowInstance.Flow(
                        i
                    ) * minCostFlowInstance.UnitCost(i)
                    print(
                        "%1s -> %1s   %3s  / %3s       %3s"
                        % (
                            minCostFlowInstance.Tail(i),
                            minCostFlowInstance.Head(i),
                            minCostFlowInstance.Flow(i),
                            minCostFlowInstance.Capacity(i),
                            cost,
                        )
                    )
            return {
                "score": minCostFlowInstance.OptimalCost()
                + create_destroy_cost,
                "minCostFlow": minCostFlowInstance,
                "mass_create_destroy_cost": create_destroy_cost,
            }
        print(
            "There was an issue with the input.\n Note: .SolveMaxFlowWithMinCost tried"
        )


def print_flow(minCostFlowInstance):
    "print the flow after calculation"
    for i in range(minCostFlowInstance.NumArcs()):
        cost = minCostFlowInstance.Flow(i) * minCostFlowInstance.UnitCost(i)
        print(
            "%1s -> %1s   %3s  / %3s       %3s"
            % (
                minCostFlowInstance.Tail(i),
                minCostFlowInstance.Head(i),
                minCostFlowInstance.Flow(i),
                minCostFlowInstance.Capacity(i),
                cost,
            )
        )


def calc_emdiv(
    edge_dict, node_supplies_dict, extra_cost_per_unit, print_out=True
):
    """
    Calculate Earth Mover Divergence (EMDiv)
    For EMDiv, the sum of node_supplies_dict must be equal to zero
    (supply equals demand). This can be done by normalizing predictions or
    actuals. This function can be used with or without cost or edge asymmetries.
    Inputs:
       edge_dict: usually from make_edges_dict, includes key, value pairs
           "edge_heads": A list of all node heads for edges, integers starting at 0, ints
           "edge_tails": A list of all node tails for edges, integers starting at 0, ints
           "capacities": A list of capacities for edges, must be integers, ints
           "unit_costs": A list of costs to transport 1 unit across edge, must be integers, ints
       node_supplies_dict: A dictionary of initial supplies for nodes,
           node int id as key
           supply as value, must be int.
           Note: positive is supply, negative is demand. This should be defined
               such that, for source distribution A, and target distribution B the
               supply distribution is p(A-B), with the subtraction being node by node, ints
        extra_cost_per_unit: a scalar, the cost for destroying or creating one unit, if supply
                    does not equal demand.
        print_out: print out flow, True or False (default=True)
    Output:
        a print out of results if print_out==True
        a dictionary, with keys:
            score: the emdiv score, in units of work (lower is better)
            minCostFlow: the minCostFlowInstance object
            massCreatDestroyCost: the total costs of mass creation and destruction
    Note: Always calls .Solve since it is assumed
        np.sum([j for i,j in node_supplies_dict.items()])==0
    """
    # check that supplies sum to zero since that is neccesary for emdiv
    # (but not pemdiv)
    try:
        check_sum_to_0(node_supplies_dict)
    except SupplyDemandMismatchError as err:
        print(err + " You might look at calc_pemdiv.")
        return {}
    # Call is just to calc_pemdiv with solver="minCost" which calls .Solve
    calc_pemdiv(
        edge_dict=edge_dict,
        node_supplies_dict=node_supplies_dict,
        extra_cost_per_unit=extra_cost_per_unit,
        solver="minCost",
        print_out=print_out,
    )


def make_edgesDict_from_df_panel(
    dist_mat,
    df,
    row_col_names,
    sloc_name_col,
    tloc_col,
    node_id_col,
    default_capacity=200000,
    time_cost=1,
):
    """
    go from a dist_mat (must be specified, even if None) and a pd data frame (cross-sectional time series) to a edgesDict obj
        Note: if dist_mat=None then only time connections are created
    Input:
        dist_mat --- square 2-d np.array with 0s for no edges and non-zeros as edges (row is heads
            column is tail of edge) or None; if dist_mat=None, then only time connections are considered (2-d np.array)
        df --- pd.DataFrame with a column for sloc_name_col (can be None if dist_mat=None), tloc_col,
            and node_id_col (pd.DataFrame)
        row_col_names --- list of names for rows and columns in dist_mat (can be None if dist_mat=None)
        sloc_name --- name of spatial unit column in df (string)
        tloc_name --- name of temporal unit column in df (string), note values in series
            must be ints starting at 0
        node_id_col --- name of node_id_col in df (string) must be ints starting at 0.
        default_capacity --- how much mass can travel through edges (int)
        time_cost --- cost to move one unit of mass one time unit into the future
    Output:
        A dictionary of edges with keys, edge_heads, edge_tails, capacities, unit_costs
    Note:
        -- assumes time constant spatial distances in dist_mat,
        -- a spatial unit measured over time in df,
        -- that time is measured in ints starting at 0 and proceeding in sequence (discrete time)
        -- node_id should be a sequence of consequitive intergers also (starting at 0)
    """
    temp_edgeHeads = []
    temp_edgeTails = []
    temp_cost = []
    out_edgeDict = {
        "edge_heads": [],
        "edge_tails": [],
        "capacities": [],
        "unit_costs": [],
    }
    # need max time
    T = max(df[tloc_col])
    # make the time constant version
    if dist_mat is not None:
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[1]):
                if dist_mat[i, j] != 0:
                    temp_edgeHeads.append(row_col_names[i])
                    temp_edgeTails.append(row_col_names[j])
                    # note: must manually change type of dist_mat value to int from np.int64 (yes that is stupid)
                    temp_cost.append(int(dist_mat[i, j]))
        # now for each time period replicate the edges and replace with time-specific node_id names
        # count time-constant edges
        spatial_edges_num = len(temp_edgeHeads)
        T = max(df[tloc_col])
        for t in range(T):
            # subset df by t
            temp_df = df[df[tloc_col] == t]
            for e_num in range(spatial_edges_num):
                if (
                    temp_edgeHeads[e_num] in temp_df[sloc_name_col].values
                    and temp_edgeTails[e_num] in temp_df[sloc_name_col].values
                ):
                    head_node_id = temp_df[
                        temp_df[sloc_name_col] == temp_edgeHeads[e_num]
                    ][node_id_col].values.item()
                    tail_node_id = temp_df[
                        temp_df[sloc_name_col] == temp_edgeTails[e_num]
                    ][node_id_col].values.item()
                    out_edgeDict["edge_heads"].append(head_node_id)
                    out_edgeDict["edge_tails"].append(tail_node_id)
                    out_edgeDict["capacities"].append(default_capacity)
                    out_edgeDict["unit_costs"].append(temp_cost[e_num])
                else:
                    pass
    # now add the over time edges A at 0 --> A at 1, etc (subset by sloc_name),
    # for each time  t< T, check if t and t+1 is there.
    # if so, get the node ids and add them to edgeHeads, edgeTails, and then grab the capacities and costs
    for sloc in set(df[sloc_name_col]):
        temp_df = df[df[sloc_name_col] == sloc]
        for t in range(T):
            if (
                t in temp_df[tloc_col].values
                and t + 1 in temp_df[tloc_col].values
            ):
                head_node_id = temp_df[temp_df[tloc_col] == t][
                    node_id_col
                ].values.item()
                tail_node_id = temp_df[temp_df[tloc_col] == t + 1][
                    node_id_col
                ].values.item()
                out_edgeDict["edge_heads"].append(head_node_id)
                out_edgeDict["edge_tails"].append(tail_node_id)
                out_edgeDict["capacities"].append(default_capacity)
                out_edgeDict["unit_costs"].append(time_cost)
    return out_edgeDict


def add_int_node_id_to_df(df, id_name="node_id"):
    "add an integer node id to a data frame (in place)"
    if id_name in df.columns:
        print("already a column named {}; exiting".format(id_name))
        raise Exception
    df[id_name] = range(df.shape[0])
    df[id_name] = df[id_name].astype("int64")


def add_supplies_int64(
    df,
    node_id_col,
    pred_col,
    actual_col,
    int_supplies_scalar=1000,
    name_col_for_supplies="supplies",
):
    """
    Add a supplies column to a pd.DataFrame. it will be an int  (inplace operation!)
    Even if pred_col and actual_col are not
    Inputs:
        df --- the data frame to add the column (it must have a node_id_col,
            pred_col, and actual_col)
        node_id_col --- name of the column holding the nod_ids (string)
        pred_col --- name of the column holding the predictions (string)
        actual_col --- name of the column holding the actual values (string)
        int_supplies_scalar --- scalar int that will define number of significant
            digits in the integer representation of the problem; for example, if
            your smallest value of interest is .0001, then an int_supplies_scalar
            of 10000 will create an integer representation that preserves
            the significant digits (int)
        name_col_for_supplies --- the name to be used for the new column (string)
    Output:
        A pd. data from with a new column named name_col_for_supplies
    """
    if name_col_for_supplies in df.columns:
        print(
            "already a column named {}; exiting".format(name_col_for_supplies)
        )
        raise Exception
    df.sort_values(node_id_col, inplace=True)
    df[name_col_for_supplies] = (
        df[pred_col] - df[actual_col]
    ) * int_supplies_scalar
    df[name_col_for_supplies] = df[name_col_for_supplies].astype("int64")
    return df


def transmute_edges_dict_to_networkx(edges_dict):
    """
    Helper function to make our edges_dict a networkx graph
    """
    import networkx as nx

    G = nx.DiGraph()
    # create ebunches (lists of 3 -tupes, (head, tail, dict of attributes))
    temp_ebunch = [
        (edge_info[0], edge_info[1], {"unit_costs": edge_info[2]})
        for edge_info in zip(
            edges_dict["edge_heads"],
            edges_dict["edge_tails"],
            edges_dict["unit_costs"],
        )
    ]
    G.add_edges_from(temp_ebunch)
    return G


def calc_min_extra_cost_per_unit_for_panel(corner_ids, edges_dict):
    """
    Function to calculate from the minimum create/destroy cost for a given graph.
    To speed up computation, user should supply a list of all suspected corners
    (furthest points in the graph)
    Inputs:
        corner_ids --- a list of the suspected corner_ids (these are node ids) (list)
        edges_dict --- a dictionary of edges, as defined by make_edges_dict (see make_edges_dict)
    Output:
        A scalar value that is the minimum distance across the graph from all
        corners to all other corners, divided by 2. This does not have to be an int!
    Note: for time series, the corners are simply the min and max time, and the cost can
        be computed as (max-min)/2
    """
    import networkx as nx

    G = transmute_edges_dict_to_networkx(edges_dict)
    out_dist = []
    for i in corner_ids:
        for j in corner_ids:
            if i == j:
                pass
            else:
                try:
                    out_dist.append(
                        nx.dijkstra_path_length(
                            G, source=i, target=j, weight="costs"
                        )
                    )
                except nx.NetworkXNoPath:
                    # some nodes cannot be reached from other nodes... that is ok
                    pass
    return max(out_dist) / 2.0
