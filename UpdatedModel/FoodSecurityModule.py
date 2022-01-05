#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:35:49 2021

@author: Debbora Leip
"""
# get all functions
# this file is automatically sourced when running the first block of the main file

from ModelCode.Auxiliary import GetFilename

from ModelCode.CompleteModelCall import FoodSecurityProblem
from ModelCode.CompleteModelCall import LoadModelResults

from ModelCode.CropAreaComparisons import CropAreasDependingOnColaboration
from ModelCode.CropAreaComparisons import NumberOfSamples

from ModelCode.GetPenalties import GetPenalties
from ModelCode.GetPenalties import LoadPenaltyStuff

from ModelCode.GroupingClusters import GroupingClusters
from ModelCode.GroupingClusters import VisualizeClusterGroups

from ModelCode.MetaInformation import GetMetaInformation

from ModelCode.ModelCore import SolveReducedLinearProblemGurobiPy

from ModelCode.ModifySettings import ReturnGeneralSettings
from ModelCode.ModifySettings import ModifyGeneralSettings
from ModelCode.ModifySettings import ResetGeneralSettings
from ModelCode.ModifySettings import ReturnDefaultModelSettings
from ModelCode.ModifySettings import ModifyDefaultModelSettings
from ModelCode.ModifySettings import ResetDefaultModelSettings

from ModelCode.PandaGeneration import CreateEmptyPanda
from ModelCode.PandaGeneration import OpenPanda
from ModelCode.PandaGeneration import OverViewCurrentPandaVariables

from ModelCode.PandaHandling import UpdatePandaWithAddInfo
from ModelCode.PandaHandling import ReadFromPanda
from ModelCode.PandaHandling import LoadFullResults
from ModelCode.PandaHandling import RemoveRun

from ModelCode.PandaPlotsCollection import CollectionPlotsCooperationSingle
from ModelCode.PandaPlotsCollection import CollectionPlotsCooperationAgg
from ModelCode.PandaPlotsCollection import OtherPandaPlots

from ModelCode.PandaPlotFunctions import PlotPandaSingle
from ModelCode.PandaPlotFunctions import PlotPenaltyVsProb
from ModelCode.PandaPlotFunctions import PlotProbDetVsSto
from ModelCode.PandaPlotFunctions import Panda_GetResults
from ModelCode.PandaPlotFunctions import Panda_GetResultsSingScen
from ModelCode.PandaPlotFunctions import PlotPandaMedian
from ModelCode.PandaPlotFunctions import PlotPandaAll
from ModelCode.PandaPlotFunctions import PlotPandaAggregate

from ModelCode.SetFolderStructure import CheckFolderStructure
from ModelCode.SetFolderStructure import CleanFolderStructure

from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.SettingsParameters import SetParameters
from ModelCode.SettingsParameters import RiskForCatastrophe

from ModelCode.VSSandValidation import VSS
from ModelCode.VSSandValidation import OutOfSampleVal

# set up folder structure (if not already done)
CheckFolderStructure()
        