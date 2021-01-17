#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:35:49 2021

@author: Debbora Leip
"""
# get all functions 
from ModelCode.ModifyGeneralSettings import ReturnGeneralSettings
from ModelCode.ModifyGeneralSettings import ModifyGeneralSettings
from ModelCode.ModifyGeneralSettings import ResetGeneralSettings

from ModelCode.SetFolderStructure import CheckFolderStructure
from ModelCode.SetFolderStructure import CleanFolderStructure

from ModelCode.GroupingClusters import GroupingClusters
from ModelCode.GroupingClusters import VisualizeClusterGroups

from ModelCode.SettingsParameters import DefaultSettingsExcept
from ModelCode.SettingsParameters import SetParameters

from ModelCode.CompleteModelCall import FoodSecurityProblem

from ModelCode.ModelCore import SolveReducedcLinearProblemGurobiPy

from ModelCode.MetaInformation import GetMetaInformation

from ModelCode.VSSandValidation import VSS
from ModelCode.VSSandValidation import OutOfSampleVal

from ModelCode.PlottingModelOutput import PlotModelOutput

from ModelCode.AnalysisAndComparison import CompareCropAllocs
from ModelCode.AnalysisAndComparison import CompareCropAllocRiskPooling
from ModelCode.AnalysisAndComparison import GetResultsToCompare

from ModelCode.Auxiliary import filename
from ModelCode.Auxiliary import MakeList
from ModelCode.PandaFunctions import OpenPanda
from ModelCode.PandaFunctions import ReadFromPanda
from ModelCode.PandaFunctions import PandaToPlot_GetResults

# set up folder structure (if not already done)
CheckFolderStructure()
        