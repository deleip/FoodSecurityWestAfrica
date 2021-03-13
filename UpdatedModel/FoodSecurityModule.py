#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:35:49 2021

@author: Debbora Leip
"""
# get all functions 
from ModelCode.ModifySettings import ReturnGeneralSettings
from ModelCode.ModifySettings import ModifyGeneralSettings
from ModelCode.ModifySettings import ResetGeneralSettings
from ModelCode.ModifySettings import ReturnDefaultModelSettings
from ModelCode.ModifySettings import ModifyDefaultModelSettings
from ModelCode.ModifySettings import ResetDefaultModelSettings

from ModelCode.SetFolderStructure import CheckFolderStructure
from ModelCode.SetFolderStructure import CleanFolderStructure

from ModelCode.GroupingClusters import GroupingClusters
from ModelCode.GroupingClusters import VisualizeClusterGroups

from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.SettingsParameters import SetParameters

from ModelCode.CompleteModelCall import FoodSecurityProblem
from ModelCode.CompleteModelCall import LoadModelResults
from ModelCode.CompleteModelCall import PlotCropAlloc

from ModelCode.ModelCore import SolveReducedcLinearProblemGurobiPy

from ModelCode.MetaInformation import GetMetaInformation

from ModelCode.VSSandValidation import VSS
from ModelCode.VSSandValidation import OutOfSampleVal

from ModelCode.CropAreaComparisons import CompareCropAllocs
from ModelCode.CropAreaComparisons import CompareCropAllocRiskPooling
from ModelCode.CropAreaComparisons import GetResultsToCompare
from ModelCode.CropAreaComparisons import CropAreasDependingOnColaboration
from ModelCode.CropAreaComparisons import NumberOfSamples

from ModelCode.Auxiliary import GetFilename
from ModelCode.Auxiliary import MakeList
from ModelCode.PandaGeneration import OpenPanda
from ModelCode.PandaGeneration import SetUpNewCurrentPandas
from ModelCode.PandaGeneration import OverViewCurrentPandaVariables
from ModelCode.PandaHandling import UpdatePandaWithAddInfo
from ModelCode.PandaHandling import ReadFromPanda
from ModelCode.PandaPlotsSingleScenario import PandaToPlot_GetResultsSingScen
from ModelCode.PandaPlotsSingleScenario import PlotPandaMedian
from ModelCode.PandaPlotsSingleScenario import PlotPandaAll
from ModelCode.PandaPlotsSingleScenario import PlotPandaAggregate
from ModelCode.PandaPlotsSingleScenario import PandaPlotsCooperation
from ModelCode.PandaPlotsSingleScenario import OtherPandaPlots

# set up folder structure (if not already done)
CheckFolderStructure()
        