# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:05:06 2022

@author: leip
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:28:30 2021

@author: Debbora Leip
"""
import sys

from ModelCode.PandaPlotFunctions import PlotPandaSingle
from ModelCode.PandaPlotFunctions import PlotPandaAggregate
from ModelCode.PandaPlotFunctions import PlotPenaltyVsProb
from ModelCode.PandaPlotFunctions import PlotProbDetVsSto
from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.Auxiliary import GetFilename
from ModelCode.SetFolderStructure import _GroupingPlotFolders

# %% ############### PLOTTING FUNCTIONS USING RESULTS PANDA CSV ###############


def CollectionPlotsCooperationSingle(panda_file = "current_panda", 
                                     scenarionames = None,
                                     folder_comparisons = "unnamed",
                                     publication_plot = False,
                                     fn_suffix = None,
                                     grouping_aim = "Dissimilar",
                                     grouping_metric = "medoids",
                                     adjacent = False,
                                     plt_legend = True,
                                     close_plots = None,
                                     console_output = None,
                                     figsize = None,
                                     **kwargs):
    """
    
    Collection of plotting function calls for variables shown as for each 
    cluster group separately. 
    
    Parameters
    ----------
    panda_file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    scenarionames : list of str, optional
        Added as legend to describe the different scenarios, and leads to plots
        being saved in /ComparingScenarios. If None, the folder according
        grouping_aim and adjacent is used. Default is None.
    folder_comparisons: str
        Subfolder of /ComparingScenarios (i.e. for example
        ComparisonPlots/folder_comparison/AggregatedSum/NecImport.png).
        Only relevant if scenarionames is not None. Default is "unnamed".
    publication_plot : boolean
        Whether to save the scenario comparison plots in the folde for the
        final publication plots, or in the normal folder for comparisons.
        Default is False.
    fn_suffix : str, optional
        Suffix to add to filename (normally defining the settings for which 
        model results are visualized). Default is None. 
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
    adjacent : boolean, optional
        Whether clusters in a cluster group need to be adjacent. The default is False.
    plt_legend : boolean
        Whether legend should be plotted (in case with multiple scenarios).
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.        
    figsize : tuple
        Size of output figures. If None, the default as defined in
        ModelCode/GeneralSettings is used.
    **kwargs : 
        Settings specifiying for which model runs we want the plots
    
    Returns
    -------
    None.

    """
    # settings
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
    
    if scenarionames is None:
        foldername = grouping_aim
        if adjacent:
            foldername = foldername + "Adjacent"
        else:
            foldername = foldername + "NonAdjacent"
        foldername = foldername + "/PandaPlots"
    else:
        if publication_plot:
            _GroupingPlotFolders(main = "PublicationPlots/" + folder_comparisons, a = False)
            foldername = "PublicationPlots/" + folder_comparisons
        else:
            _GroupingPlotFolders(main = "ComparingScenarios/" + folder_comparisons, a = False)
            foldername = "ComparingScenarios/" + folder_comparisons
        
    if fn_suffix is None:
        settingsIterate = DefaultSettingsExcept(**kwargs)
        settingsIterate["N"] = ""
        settingsIterate["validation_size"] = ""
        settingsIterate["k_using"] = ""
        fn_suffix = "_" + GetFilename(settingsIterate, groupSize = "", groupAim = grouping_aim, \
                          adjacent = adjacent)
            
            
    def _report(i, console_output = console_output, num_plots = 5):
        if console_output:
            sys.stdout.write("\r     Plot " + str(i) + " of " + str(num_plots))
        return(i + 1)
    i = 1         
    
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Penalty for food shortage', \
                                'Penalty for insolvency'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "Penalties" + fn_suffix,
                    foldername = foldername,
                    plt_legend = plt_legend,
                    close_plots = close_plots,
                    figsize = figsize,
                    **kwargs)
    i = _report(i)    

    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Resulting probability for food security', \
                                'Resulting probability for solvency'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "ResProbabilities" + fn_suffix,
                    foldername = foldername,
                    plt_legend = plt_legend,
                    close_plots = close_plots,
                    figsize = figsize,
                    **kwargs)
    i = _report(i)    
 
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Average aggregate food shortage per capita', \
                                'Average aggregate debt after payout per capita'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "ShortcomingsCapita" + fn_suffix,
                    foldername = foldername,
                    plt_legend = plt_legend,
                    close_plots = close_plots,
                    figsize = figsize,
                    **kwargs)
    i = _report(i)  
    
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Average aggregate food shortage per capita (including only samples that have shortage)', \
                                'Average aggregate debt after payout per capita (including only samples with negative final fund)'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "ShortcomingsOnlyWhenNeededCapita" + fn_suffix,
                    foldername = foldername,
                    plt_legend = plt_legend,
                    close_plots = close_plots,
                    figsize = figsize,
                    **kwargs)
    i = _report(i)  
    
    PlotPandaSingle(panda_file = panda_file,
                    output_var=['Average aggregate food shortage (without taking into account imports)', \
                                'Average aggregate debt after payout'],
                    scenarionames = scenarionames,
                    grouping_aim = grouping_aim,
                    grouping_metric = grouping_metric,
                    adjacent = adjacent,
                    plt_file = "Shortcomings" + fn_suffix,
                    foldername = foldername,
                    plt_legend = plt_legend,
                    close_plots = close_plots,
                    figsize = figsize,
                    **kwargs)
    i = _report(i) 
        
    # will not talk about technical VSS
    # PlotPandaSingle(panda_file = panda_file,
    #                 output_var=['Value of stochastic solution', \
    #                             'VSS as share of total costs (sto. solution)',\
    #                             'VSS as share of total costs (det. solution)'],
    #                 scenarionames = scenarionames,
    #                 grouping_aim = grouping_aim,
    #                 grouping_metric = grouping_metric,
    #                 adjacent = adjacent,
    #                 plt_file = "VSScosts" + fn_suffix,
    #                 foldername = foldername,
    #                 plt_legend = plt_legend,
    #                 close_plots = close_plots,
    #                   figsize = figsize,
    #                 **kwargs)
    # i = _report(i)    
    
    # will not talk about technical VSS
    # PlotPandaSingle(panda_file = panda_file,
    #                 output_var=['VSS in terms of avg. nec. debt', \
    #                             'VSS in terms of avg. nec. debt as share of avg. nec. debt of det. solution',\
    #                             'VSS in terms of avg. nec. debt as share of avg. nec. debt of sto. solution'],
    #                 scenarionames = scenarionames,
    #                 grouping_aim = grouping_aim,
    #                 grouping_metric = grouping_metric,
    #                 adjacent = adjacent,
    #                 plt_file = "VSSdebt" + fn_suffix,
    #                 foldername = foldername,
    #                 plt_legend = plt_legend,
    #                 close_plots = close_plots,
    #                   figsize = figsize,
    #                 **kwargs)
    # i = _report(i)  
    
    # will not talk about technical VSS
    # PlotPandaSingle(panda_file = panda_file,
    #                 output_var=['VSS in terms of avg. nec. import', \
    #                             'VSS in terms of avg. nec. import as share of avg. nec. import of det. solution',\
    #                             'VSS in terms of avg. nec. import as share of avg. nec. import of sto. solution'],
    #                 scenarionames = scenarionames,
    #                 grouping_aim = grouping_aim,
    #                 grouping_metric = grouping_metric,
    #                 adjacent = adjacent,
    #                 plt_file = "VSSimport" + fn_suffix,
    #                 foldername = foldername,
    #                 plt_legend = plt_legend,
    #                 close_plots = close_plots,
    #                   figsize = figsize,
    #                 **kwargs)
    # i = _report(i)  
    
    # will not talk about technical VSS
    # PlotPandaSingle(panda_file = panda_file,
    #                 output_var=['Resulting probability for food security for VSS',\
    #                             'Resulting probability for solvency for VSS'],
    #                 scenarionames = scenarionames,
    #                 grouping_aim = grouping_aim,
    #                 grouping_metric = grouping_metric,
    #                 adjacent = adjacent,
    #                 plt_file = "VSSprobabilities" + fn_suffix,
    #                 foldername = foldername,
    #                 plt_legend = plt_legend,
    #                 close_plots = close_plots,
    #                   figsize = figsize,
    #                 **kwargs)
    # i = _report(i)    
    
    
    return(None)


def CollectionPlotsCooperationAgg(panda_file = "current_panda", 
                                  scenarionames = None,
                                  scenarios_shaded = False,
                                  folder_comparisons = "unnamed",
                                  publication_plot = False,
                                  fn_suffix = None,
                                  grouping_aim = "Dissimilar",
                                  grouping_metric = "medoids",
                                  adjacent = False,
                                  plt_legend = True,
                                  close_plots = None,
                                  console_output = None,
                                  figsize = None,
                                  **kwargs):
    """
    
    Collection of plotting function calls for variables shown as aggregate
    over all cluster groups.

    Parameters
    ----------
    panda_file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    scenarionames : list of str, optional
        Added as legend to describe the different scenarios, and leads to plots
        being saved in /ComparingScenarios. If None, the folder according
        grouping_aim and adjacent is used. Default is None.
    scenarios_shaded : boolean, optional
        Whether area between two scenarios should be shaded. To be used if
        worst and best case scenario is to be compared for different settings.
        In this case, an even number of scenarios must be given, and always 
        two sequential scenarios are seen as a pair and the area between them
        will be shaded.
    folder_comparisons: str
        Subfolder of /ComparingScenarios (i.e. for example
        ComparisonPlots/folder_comparison/AggregatedSum/NecImport.png).
        Only relevant if scenarionames is not None. Default is "unnamed".
    publication_plot : boolean
        Whether to save the scenario comparison plots in the folde for the
        final publication plots, or in the normal folder for comparisons.
        Default is False.
    fn_suffix : str, optional
        Suffix to add to filename (normally defining the settings for which 
        model results are visualized). Default is None.
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
    adjacent : boolean, optional
        Whether clusters in a cluster group need to be adjacent. The default is False.
    plt_legend : boolean
        Whether legend should be plotted (in case with multiple scenarios).
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.        
    figsize : tuple
        Size of output figures. If None, the default as defined in
        ModelCode/GeneralSettings is used.
    **kwargs : 
        Settings specifiying for which model runs we want the plots
    
    Returns
    -------
    None.

    """
    # settings
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
    
    if scenarionames is None:
        foldername = grouping_aim
        if adjacent:
            foldername = foldername + "Adjacent"
        else:
            foldername = foldername + "NonAdjacent"
        foldername = foldername + "/PandaPlots"
    else:
        if publication_plot:
            _GroupingPlotFolders(main = "PublicationPlots/" + folder_comparisons, a = False)
            foldername = "PublicationPlots/" + folder_comparisons
        else:
            _GroupingPlotFolders(main = "ComparingScenarios/" + folder_comparisons, a = False)
            foldername = "ComparingScenarios/" + folder_comparisons
        
    if fn_suffix is None:
        settingsIterate = DefaultSettingsExcept(**kwargs)
        settingsIterate["N"] = ""
        settingsIterate["validation_size"] = ""
        settingsIterate["k_using"] = ""
        fn_suffix = "_" + GetFilename(settingsIterate, groupSize = "", groupAim = grouping_aim, \
                          adjacent = adjacent)
            
            
    def _report(i, console_output = console_output, num_plots = 12):
        if console_output:
            sys.stdout.write("\r     Plot " + str(i) + " of " + str(num_plots))
        return(i + 1)
    i = 1         
    
    # plotting:
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_sum",
                       output_var=['Average yearly total cultivated area'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "TotalAllocArea" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)    
    
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_sum",
                       output_var=['Average yearly total cultivated area'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       scale_by = "Available arable area",
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "TotalAllocArea_ScaledByTotalArable" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)    
        
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_sum",
                       output_var=['Total cultivation costs (sto. solution)'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "TotalCultCosts" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)    
    
    # related to objective function
    # PlotPandaAggregate(panda_file = panda_file,
    #                    agg_type = "agg_sum",
    #                    output_var=["Average total cultivation costs", \
    #                                "Average total food demand penalty (over samples)", \
    #                                "Average solvency penalty (over samples)"],
    #                    scenarionames = scenarionames,
    #                    scenarios_shaded = scenarios_shaded,
    #                    grouping_aim = grouping_aim,
    #                    grouping_metric = grouping_metric,
    #                    adjacent = adjacent,
    #                    plt_file = "CultivationAndSocialCosts" + fn_suffix,
    #                    foldername = foldername,
    #                    plt_legend = plt_legend,
    #                    close_plots = close_plots,
    #                    **kwargs)
    # i = _report(i)    
    
    # we don't have a solvency contraint anymore...
    # PlotPandaAggregate(panda_file = panda_file,
    #                    agg_type = "agg_sum", 
    #                    output_var=['Average aggregate food shortage excluding solvency constraint', \
    #                                'Average aggregate debt after payout (excluding food security constraint)'],
    #                    scenarionames = scenarionames,
    #                    scenarios_shaded = scenarios_shaded,
    #                    grouping_aim = grouping_aim,
    #                    grouping_metric = grouping_metric,
    #                    adjacent = adjacent,
    #                    plt_file = "NecImportsPen_NecDebtPen" + fn_suffix,
    #                    foldername = foldername,
    #                    plt_legend = plt_legend,
    #                    close_plots = close_plots,
    #                    **kwargs)
    # i = _report(i)    
        
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_sum",
                       output_var=['Average aggregate food shortage'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "AggFoodShortage" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)    
  
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_sum",
                       output_var=['Average aggregate debt after payout'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "AggDebt" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)    
    
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       weight_title = "population",
                       output_var=['Resulting probability for food security'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "ResFoodSecProb" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)    

    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       weight_title = "population",
                       output_var=['Resulting probability for solvency'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "ResSolvProb" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)    

    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       weight_title = "population",
                       output_var=['Average aggregate food shortage per capita'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "FoodShortcomingsCapita" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)   
        
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       weight_title = "population",
                       output_var=['Average aggregate food shortage per capita (including only samples that have shortage)'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "FoodShortcomingsCapita" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)   
    
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       weight_title = "population",
                       output_var=['Average aggregate debt after payout per capita'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "DebtCapita" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)   
       
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       weight_title = "population",
                       output_var=['Average aggregate debt after payout per capita (including only samples with negative final fund)'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "DebtCapitaNegFund" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)   
         
    PlotPandaAggregate(panda_file = panda_file,
                       agg_type = "agg_avgweight",
                       var_weight = "Share of West Africa's population that is living in total considered region (2015)",
                       weight_title = "population",
                       output_var=['Average aggregate debt after payout per capita (including only samples with catastrophe)'],
                       scenarionames = scenarionames,
                       scenarios_shaded = scenarios_shaded,
                       grouping_aim = grouping_aim,
                       grouping_metric = grouping_metric,
                       adjacent = adjacent,
                       plt_file = "DebtCapitaOnlyCatastrophe" + fn_suffix,
                       foldername = foldername,
                       plt_legend = plt_legend,
                       close_plots = close_plots,
                       figsize = figsize,
                       **kwargs)
    i = _report(i)   
    # will not talk about penalties explicitly
    # PlotPandaAggregate(panda_file = panda_file,
    #                    agg_type = "agg_sum", 
    #                    output_var=['Average food demand penalty (over samples and then years)', \
    #                                'Average solvency penalty (over samples)'],
    #                    scenarionames = scenarionames,
    #                    scenarios_shaded = scenarios_shaded,
    #                    grouping_aim = grouping_aim,
    #                    grouping_metric = grouping_metric,
    #                    adjacent = adjacent,
    #                    plt_file = "PenaltiesPaied" + fn_suffix,
    #                    foldername = foldername,
    #                    plt_legend = plt_legend,
    #                    close_plots = close_plots,
    #                    **kwargs)
    # i = _report(i)    
    
    # will not talk about technical VSS
    # PlotPandaAggregate(panda_file = panda_file,
    #                    agg_type = "agg_sum", 
    #                    output_var=['Value of stochastic solution', \
    #                                'VSS in terms of avg. nec. debt', \
    #                                'VSS in terms of avg. nec. import'],
    #                    scenarionames = scenarionames,
    #                    scenarios_shaded = scenarios_shaded,
    #                    grouping_aim = grouping_aim,
    #                    grouping_metric = grouping_metric,
    #                    adjacent = adjacent,
    #                    plt_file = "VSSagg" + fn_suffix,
    #                    foldername = foldername,
    #                    plt_legend = plt_legend,
    #                    close_plots = close_plots,
    #                    **kwargs)
    
    return(None)


def OtherPandaPlots(panda_file = "current_panda", 
                    grouping_aim = "Dissimilar",
                    grouping_metric = "medoids",
                    adjacent = False,
                    close_plots = None,
                    console_output = None,
                    fn_suffix = None,
                    **kwargs):
    """
    Creates some additional plots (that don't fit into the structure of 
    PandaPlotsCooperation): PlotPenaltyVsProb and PlotProbDetVsSto

    Parameters
    ----------
    panda_file : str, optional
        Filename of the panda csv to use. The default is "current_panda".
    grouping_aim : str, optional
        The aim in grouping clusters, either "Similar" or "Dissimilar".
        The default is "Dissimilar".
    grouping_metric : str, optional
        The metric on which the grouping is based. The default is "medoids".
    adjacent : boolean, optional
        Whether clusters in a cluster group need to be adjacent. The default is False.
    close_plots : boolean or None
        Whether plots should be closed after plotting (and saving). If None, 
        the default as defined in ModelCode/GeneralSettings is used.
    console_output : boolean, optional
        Specifying whether the progress should be documented thorugh console 
        outputs. If None, the default as defined in ModelCode/GeneralSettings 
        is used.        
    fn_suffix : str, optional
        Suffix to add to filename (normally defining the settings for which 
        model results are visualized). Default is None.
    **kwargs : 
        Settings specifiying for which model runs we want the plots 

    Returns
    -------
    None.

    """
    
    # settings
    if console_output is None:
        from ModelCode.GeneralSettings import console_output
      
    # foldername = grouping_aim
    # if adjacent:
    #     foldername = foldername + "Adjacent"
    # else:
    #     foldername = foldername + "NonAdjacent"
    # foldername = foldername + "/PandaPlots"
        
    if fn_suffix is None:
        settingsIterate = DefaultSettingsExcept(**kwargs)
        settingsIterate["N"] = ""
        settingsIterate["validation_size"] = ""
        settingsIterate["k_using"] = ""
        fn_suffix = "_" + GetFilename(settingsIterate, groupSize = "", groupAim = grouping_aim, \
                          adjacent = adjacent)
            
    # plot penalties vs. probabilities
    PlotPenaltyVsProb(panda_file = panda_file, 
                  grouping_aim = grouping_aim,
                  grouping_metric = grouping_metric,
                  adjacent = adjacent,
                  close_plots = close_plots,
                  fn_suffix = fn_suffix, 
                  **kwargs)
    
    # plot sto. probabilities vs. det. probabilities
    PlotProbDetVsSto(panda_file = panda_file, 
                     grouping_aim = grouping_aim,
                     grouping_metric = grouping_metric,
                     adjacent = adjacent,
                     close_plots = close_plots,
                     fn_suffix = fn_suffix, 
                     **kwargs)
    return(None)
