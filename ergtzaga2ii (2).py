import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')


def matmultiply(mat1,mat2):
    
    return( np.matmul(mat1, mat2) )
    
def calculateCost(indV, depV, thetas):
    return( np.sum( ((matmultiply(indV, thetas) - depV)**2) / (2*indV.shape[0]) ) )  
    

def batchGradientDescent(indV, depV, thetas, alpha=0.1, numIters=200, verbose=False):
    calcThetas = thetas
    costHistory = pd.DataFrame(columns=["iter", "cost"])
    m = len(depV)
    for i in range(0, numIters):
        prediction = np.dot(indV, calcThetas)
        calcThetas = calcThetas - (1 / m) * alpha * (indV.T.dot(prediction - depV))
        print(">>>> Iteration", i, ")")
        print("       Calculate thetas...", calcThetas)
        c = calculateCost(indV, depV, calcThetas)
        print("       Calculate cost fuction for new thetas...", c)
        costHistory = costHistory.append({"iter": i, "cost": c}, ignore_index=True)
    return calcThetas, costHistory



communities = pd.read_csv("communities.data", header=None, sep=",", engine='python')
communities = communities.set_axis(["state", "county", "community", "communityname", "fold", "population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian",
                           "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWage", "pctWFarmSelf", 
                           "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc",
                           "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade",
                           "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce",
                           "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids",
                           "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
                           "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous",
                           "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", 
                           "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian",
                           "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState",
                           "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq", 
                           "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", 
                           "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg", 
                           "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop", "ViolentCrimesPerPop"], axis=1)

dependentVar = communities.iloc[:, 127]

independentVars = communities.iloc[:, [17,26,27,31,32,37,76,90,95] ]

independentVars = independentVars[(independentVars != '?').all(1)]

independentVars.insert(0, 'b0', 1)


iniThetas = []
for i in range(0, independentVars.shape[1]):
    iniThetas.append( np.random.rand() )

initialThetas = np.array(iniThetas)

estimatedCoefficients, costHistory = batchGradientDescent(independentVars.to_numpy(), dependentVar.to_numpy(), initialThetas, 0.1, 200)

print(estimatedCoefficients)

costHistory.plot.scatter(x="iter", y="cost", color='red')
plt.show()


