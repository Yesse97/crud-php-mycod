# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 21:05:36 2024

@author: Hp
"""

#IMPORTAR LIBRERIAS
import pandas as pd
import numpy as np 
import seaborn as sns
pip install yfinance
import yfinance as yf

#SUBIR BASE DE DATOS 
GCO_df =  yf.download("GCO", start= "2024-01-01", end="2024-07-01")
GCO= pd.DataFrame(GCO_df["Adj Close"])
GCO.columns=["GCO"] 

NVDA_df =  yf.download("NVDA", start= "2024-01-01", end="2024-07-01")
NVDA= pd.DataFrame(NVDA_df["Adj Close"])
NVDA.columns=["NVDA"]

CXW_df =  yf.download("CXW", start= "2024-01-01", end="2024-07-01")
CXW= pd.DataFrame(CXW_df["Adj Close"])
CXW.columns=["CXW"]

EAT_df =  yf.download("EAT", start= "2024-01-01", end="2024-07-01")
EAT= pd.DataFrame(EAT_df["Adj Close"])
EAT.columns=["EAT"]

OSIS_df =  yf.download("OSIS", start= "2024-01-01", end="2024-07-01")
OSIS= pd.DataFrame(OSIS_df["Adj Close"])
OSIS.columns=["OSIS"]

KMT_df =  yf.download("KMT", start= "2024-01-01", end="2024-07-01")
KMT= pd.DataFrame(KMT_df["Adj Close"])
KMT.columns=["KMT"]

DAN_df =  yf.download("KMT", start= "2024-01-01", end="2024-07-01")
DAN= pd.DataFrame(DAN_df["Adj Close"])
DAN.columns=["DAN"]


EBF_df =  yf.download("EBF", start= "2024-01-01", end="2024-07-01")
EBF= pd.DataFrame(EBF_df["Adj Close"])
EBF.columns=["EBF"]

SCS_df =  yf.download("SCS", start= "2024-01-01", end="2024-07-01")
SCS= pd.DataFrame(SCS_df["Adj Close"])
SCS.columns=["SCS"]

MLKN_df =  yf.download("MLKN", start= "2024-01-01", end="2024-07-01")
MLKN= pd.DataFrame(MLKN_df["Adj Close"])
MLKN.columns=["MLKN"]

TILE_df =  yf.download("TILE", start= "2024-01-01", end="2024-07-01")
TILE= pd.DataFrame(TILE_df["Adj Close"])
TILE.columns=["TILE"]


QUAD_df =  yf.download("QUAD", start= "2024-01-01", end="2024-07-01")
QUAD= pd.DataFrame(QUAD_df["Adj Close"])
QUAD.columns=["QUAD"]


CZNC_df =  yf.download("CZNC", start= "2024-01-01", end="2024-07-01")
CZNC= pd.DataFrame(CZNC_df["Adj Close"])
CZNC.columns=["CZNC"]


MRO_df =  yf.download("MRO", start= "2024-01-01", end="2024-07-01")
MRO= pd.DataFrame(MRO_df["Adj Close"])
MRO.columns=["MRO"]

CAC_df = yf.download("CAC", start="2024-01-01", end="2024-07-01")
CAC = pd.DataFrame(CAC_df["Adj Close"])
CAC.columns=["CAC"]

GILT_df = yf.download("GILT", start="2024-01-01", end="2024-07-01")
GILT = pd.DataFrame(GILT_df["Adj Close"])
GILT.columns=["GILT"]

WAFD_df = yf.download("WAFD", start="2024-01-01", end="2024-07-01")
WAFD = pd.DataFrame(WAFD_df["Adj Close"])
WAFD.columns=["WAFD"]

GEF_df = yf.download("GEF", start="2024-01-01", end="2024-07-01")
GEF = pd.DataFrame(GEF_df["Adj Close"])
GEF.columns=["GEF"]

CMC_df = yf.download("CMC", start="2024-01-01", end="2024-07-01")
CMC = pd.DataFrame(CMC_df["Adj Close"])
CMC.columns=["CMC"]

CNOB_df = yf.download("CNOB", start="2024-01-01", end="2024-07-01")
CNOB = pd.DataFrame(CNOB_df["Adj Close"])
CNOB.columns=["CNOB"]

FULT_df = yf.download("FULT", start="2024-01-01", end="2024-07-01")
FULT = pd.DataFrame(FULT_df["Adj Close"])
FULT.columns=["FULT"]

KBH_df = yf.download("KBH", start="2024-01-01", end="2024-07-01")
KBH = pd.DataFrame(KBH_df["Adj Close"])
KBH.columns=["KBH"]

HVT_df = yf.download("HVT", start="2024-01-01", end="2024-07-01")
HVT = pd.DataFrame(HVT_df["Adj Close"])
HVT.columns=["HVT"]

PEBO_df = yf.download("PEBO", start="2024-01-01", end="2024-07-01")
PEBO = pd.DataFrame(PEBO_df["Adj Close"])
PEBO.columns=["PEBO"]

FFBC_df = yf.download("FFBC", start="2024-01-01", end="2024-07-01")
FFBC = pd.DataFrame(FFBC_df["Adj Close"])
FFBC.columns=["FFBC"]

MPC_df = yf.download("MPC", start="2024-01-01", end="2024-07-01")
MPC = pd.DataFrame(MPC_df["Adj Close"])
MPC.columns=["MPC"]

NMIH_df = yf.download("NMIH", start="2024-01-01", end="2024-07-01")
NMIH = pd.DataFrame(NMIH_df["Adj Close"])
NMIH.columns=["NMIH"]

CATY_df = yf.download("CATY", start="2024-01-01", end="2024-07-01")
CATY = pd.DataFrame(CATY_df["Adj Close"])
CATY.columns=["CATY"]

MTG_df = yf.download("MTG", start="2024-01-01", end="2024-07-01")
MTG = pd.DataFrame(MTG_df["Adj Close"])
MTG.columns=["MTG"]

RDN_df = yf.download("RDN", start="2024-01-01", end="2024-07-01")
RDN = pd.DataFrame(RDN_df["Adj Close"])
RDN.columns=["RDN"]

DAL_df = yf.download("DAL", start="2024-01-01", end="2024-07-01")
DAL = pd.DataFrame(DAL_df["Adj Close"])
DAL.columns=["DAL"]

F_df = yf.download("F", start="2024-01-01", end="2024-07-01")
F = pd.DataFrame(F_df["Adj Close"])
F.columns=["F"]

NOG_df = yf.download("NOG", start="2024-01-01", end="2024-07-01")
NOG = pd.DataFrame(NOG_df["Adj Close"])
NOG.columns=["NOG"]

BANC_df = yf.download("BANC", start="2024-01-01", end="2024-07-01")
BANC = pd.DataFrame(BANC_df["Adj Close"])
BANC.columns=["BANC"]

GM_df = yf.download("GM", start="2024-01-01", end="2024-07-01")
GM = pd.DataFrame(GM_df["Adj Close"])
GM.columns=["GM"]

WNC_df = yf.download("WNC", start="2024-01-01", end="2024-07-01")
WNC = pd.DataFrame(WNC_df["Adj Close"])
WNC.columns=["WNC"]

UAL_df = yf.download("UAL", start="2024-01-01", end="2024-07-01")
UAL = pd.DataFrame(UAL_df["Adj Close"])
UAL.columns=["UAL"]

VTLE_df = yf.download("VTLE", start="2024-01-01", end="2024-07-01")
VTLE = pd.DataFrame(VTLE_df["Adj Close"])
VTLE.columns=["VTLE"]

TSE_df = yf.download("TSE", start="2024-01-01", end="2024-07-01")
TSE = pd.DataFrame(TSE_df["Adj Close"])
TSE.columns=["TSE"]

TPC_df = yf.download("TPC", start="2024-01-01", end="2024-07-01")
TPC = pd.DataFrame(TPC_df["Adj Close"])
TPC.columns=["TPC"]

PDM_df = yf.download("PDM", start="2024-01-01", end="2024-07-01")
PDM = pd.DataFrame(PDM_df["Adj Close"])
PDM.columns=["PDM"]

VVX_df = yf.download("VVX", start="2024-01-01", end="2024-07-01")
VVX = pd.DataFrame(VVX_df["Adj Close"])
VVX.columns=["VVX"]

IMKTA_df = yf.download("IMKTA", start="2024-01-01", end="2024-07-01")
IMKTA = pd.DataFrame(IMKTA_df["Adj Close"])
IMKTA.columns=["IMKTA"]

FNLC_df = yf.download("FNLC", start="2024-01-01", end="2024-07-01")
FNLC = pd.DataFrame(FNLC_df["Adj Close"])
FNLC.columns=["FNLC"]

# 2.2. Rendimiento y riesgo 
GCO_var = GCO.pct_change()
GCO_logf = np.log(GCO).diff()
GCO_var = GCO_var.dropna()
GCO_logf = GCO_logf.dropna() 
GCO_rend = np.mean(GCO_logf)
GCO_std = GCO_logf.std()
print(GCO_rend)
print(GCO_std)

NVDA_var = NVDA.pct_change()
NVDA_logf = np.log(NVDA).diff()
NVDA_var = NVDA_var.dropna()
NVDA_logf = NVDA_logf.dropna() 
NVDA_rend = np.mean(NVDA_logf)
NVDA_std = NVDA_logf.std()
print(NVDA_rend)
print(NVDA_std)


CXW_var = CXW.pct_change()
CXW_logf = np.log(CXW).diff()
CXW_var = CXW_var.dropna()
CXW_logf = CXW_logf.dropna() 
CXW_rend = np.mean(CXW_logf)
CXW_std = CXW_logf.std()
print(CXW_rend)
print(CXW_std)

EAT_var = EAT.pct_change()
EAT_logf = np.log(EAT).diff()
EAT_var = EAT_var.dropna()
EAT_logf = EAT_logf.dropna() 
EAT_rend = np.mean(EAT_logf)
EAT_std = EAT_logf.std()
print(EAT_rend)
print(EAT_std)

OSIS_var = OSIS.pct_change()
OSIS_logf = np.log(OSIS).diff()
OSIS_var = OSIS_var.dropna()
OSIS_logf = OSIS_logf.dropna() 
OSIS_rend = np.mean(OSIS_logf)
OSIS_std = OSIS_logf.std()
print(OSIS_rend)
print(OSIS_std)

KMT_var = KMT.pct_change()
KMT_logf = np.log(KMT).diff()
KMT_var = KMT_var.dropna()
KMT_logf = KMT_logf.dropna() 
KMT_rend = np.mean(KMT_logf)
KMT_std = KMT_logf.std()
print(KMT_rend)
print(KMT_std)


DAN_var = DAN.pct_change()
DAN_logf = np.log(DAN).diff()
DAN_var = DAN_var.dropna()
DAN_logf = DAN_logf.dropna()
DAN_rend = np.mean(DAN_logf)
DAN_std = DAN_logf.std()
print(DAN_rend)
print(DAN_std)

EBF_var = EBF.pct_change()
EBF_logf = np.log(EBF).diff()
EBF_var = EBF_var.dropna()
EBF_logf = EBF_logf.dropna()
EBF_rend = np.mean(EBF_logf)
EBF_std = EBF_logf.std()
print(EBF_rend)
print(EBF_std)

SCS_var = SCS.pct_change()
SCS_logf = np.log(SCS).diff()
SCS_var = SCS_var.dropna()
SCS_logf = SCS_logf.dropna()
SCS_rend = np.mean(SCS_logf)
SCS_std = SCS_logf.std()
print(SCS_rend)
print(SCS_std)

MLKN_var = MLKN.pct_change()
MLKN_logf = np.log(MLKN).diff()
MLKN_var = MLKN_var.dropna()
MLKN_logf = MLKN_logf.dropna()
MLKN_rend = np.mean(MLKN_logf)
MLKN_std = MLKN_logf.std()
print(MLKN_rend)
print(MLKN_std)

TILE_var = TILE.pct_change()
TILE_logf = np.log(TILE).diff()
TILE_var = TILE_var.dropna()
TILE_logf = TILE_logf.dropna()
TILE_rend = np.mean(TILE_logf)
TILE_std = TILE_logf.std()
print(TILE_rend)
print(TILE_std)

QUAD_var = QUAD.pct_change()
QUAD_logf = np.log(QUAD).diff()
QUAD_var = QUAD_var.dropna()
QUAD_logf = QUAD_logf.dropna()
QUAD_rend = np.mean(QUAD_logf)
QUAD_std = QUAD_logf.std()
print(QUAD_rend)
print(QUAD_std)

CZNC_var = CZNC.pct_change()
CZNC_logf = np.log(CZNC).diff()
CZNC_var = CZNC_var.dropna()
CZNC_logf = CZNC_logf.dropna()
CZNC_rend = np.mean(CZNC_logf)
CZNC_std = CZNC_logf.std()
print(CZNC_rend)
print(CZNC_std)

MRO_var = MRO.pct_change()
MRO_logf = np.log(MRO).diff()
MRO_var = MRO_var.dropna()
MRO_logf = MRO_logf.dropna()
MRO_rend = np.mean(MRO_logf)
MRO_std = MRO_logf.std()
print(MRO_rend)
print(MRO_std)

CAC_var = CAC.pct_change()
CAC_logf = np.log(CAC).diff()
CAC_var = CAC_var.dropna()
CAC_logf = CAC_logf.dropna()
CAC_rend = np.mean(CAC_logf)
CAC_std = CAC_logf.std()
print(CAC_rend)
print(CAC_std)

GILT_var = GILT.pct_change()
GILT_logf = np.log(GILT).diff()
GILT_var = GILT_var.dropna()
GILT_logf = GILT_logf.dropna()
GILT_rend = np.mean(GILT_logf)
GILT_std = GILT_logf.std()
print(GILT_rend)
print(GILT_std)

WAFD_var = WAFD.pct_change()
WAFD_logf = np.log(WAFD).diff()
WAFD_var = WAFD_var.dropna()
WAFD_logf = WAFD_logf.dropna()
WAFD_rend = np.mean(WAFD_logf)
WAFD_std = WAFD_logf.std()
print(WAFD_rend)
print(WAFD_std)

GEF_var = GEF.pct_change()
GEF_logf = np.log(GEF).diff()
GEF_var = GEF_var.dropna()
GEF_logf = GEF_logf.dropna()
GEF_rend = np.mean(GEF_logf)
GEF_std = GEF_logf.std()
print(GEF_rend)
print(GEF_std)

CMC_var = CMC.pct_change()
CMC_logf = np.log(CMC).diff()
CMC_var = CMC_var.dropna()
CMC_logf = CMC_logf.dropna()
CMC_rend = np.mean(CMC_logf)
CMC_std = CMC_logf.std()
print(CMC_rend)
print(CMC_std)

CNOB_var = CNOB.pct_change()
CNOB_logf = np.log(CNOB).diff()
CNOB_var = CNOB_var.dropna()
CNOB_logf = CNOB_logf.dropna()
CNOB_rend = np.mean(CNOB_logf)
CNOB_std = CNOB_logf.std()
print(CNOB_rend)
print(CNOB_std)

FULT_var = FULT.pct_change()
FULT_logf = np.log(FULT).diff()
FULT_var = FULT_var.dropna()
FULT_logf = FULT_logf.dropna()
FULT_rend = np.mean(FULT_logf)
FULT_std = FULT_logf.std()
print(FULT_rend)
print(FULT_std)

KBH_var = KBH.pct_change()
KBH_logf = np.log(KBH).diff()
KBH_var = KBH_var.dropna()
KBH_logf = KBH_logf.dropna()
KBH_rend = np.mean(KBH_logf)
KBH_std = KBH_logf.std()
print(KBH_rend)
print(KBH_std)

HVT_var = HVT.pct_change()
HVT_logf = np.log(HVT).diff()
HVT_var = HVT_var.dropna()
HVT_logf = HVT_logf.dropna()
HVT_rend = np.mean(HVT_logf)
HVT_std = HVT_logf.std()
print(HVT_rend)
print(HVT_std)

PEBO_var = PEBO.pct_change()
PEBO_logf = np.log(PEBO).diff()
PEBO_var = PEBO_var.dropna()
PEBO_logf = PEBO_logf.dropna()
PEBO_rend = np.mean(PEBO_logf)
PEBO_std = PEBO_logf.std()
print(PEBO_rend)
print(PEBO_std)

FFBC_var = FFBC.pct_change()
FFBC_logf = np.log(FFBC).diff()
FFBC_var = FFBC_var.dropna()
FFBC_logf = FFBC_logf.dropna()
FFBC_rend = np.mean(FFBC_logf)
FFBC_std = FFBC_logf.std()
print(FFBC_rend)
print(FFBC_std)

MPC_var = MPC.pct_change()
MPC_logf = np.log(MPC).diff()
MPC_var = MPC_var.dropna()
MPC_logf = MPC_logf.dropna()
MPC_rend = np.mean(MPC_logf)
MPC_std = MPC_logf.std()
print(MPC_rend)
print(MPC_std)

NMIH_var = NMIH.pct_change()
NMIH_logf = np.log(NMIH).diff()
NMIH_var = NMIH_var.dropna()
NMIH_logf = NMIH_logf.dropna()
NMIH_rend = np.mean(NMIH_logf)
NMIH_std = NMIH_logf.std()
print(NMIH_rend)
print(NMIH_std)

CATY_var = CATY.pct_change()
CATY_logf = np.log(CATY).diff()
CATY_var = CATY_var.dropna()
CATY_logf = CATY_logf.dropna()
CATY_rend = np.mean(CATY_logf)
CATY_std = CATY_logf.std()
print(CATY_rend)
print(CATY_std)

MTG_var = MTG.pct_change()
MTG_logf = np.log(MTG).diff()
MTG_var = MTG_var.dropna()
MTG_logf = MTG_logf.dropna()
MTG_rend = np.mean(MTG_logf)
MTG_std = MTG_logf.std()
print(MTG_rend)
print(MTG_std)

RDN_var = RDN.pct_change()
RDN_logf = np.log(RDN).diff()
RDN_var = RDN_var.dropna()
RDN_logf = RDN_logf.dropna()
RDN_rend = np.mean(RDN_logf)
RDN_std = RDN_logf.std()
print(RDN_rend)
print(RDN_std)

DAL_var = DAL.pct_change()
DAL_logf = np.log(DAL).diff()
DAL_var = DAL_var.dropna()
DAL_logf = DAL_logf.dropna()
DAL_rend = np.mean(DAL_logf)
DAL_std = DAL_logf.std()
print(DAL_rend)
print(DAL_std)

F_var = F.pct_change()
F_logf = np.log(F).diff()
F_var = F_var.dropna()
F_logf = F_logf.dropna()
F_rend = np.mean(F_logf)
F_std = F_logf.std()
print(F_rend)

NOG_var = NOG.pct_change()
NOG_logf = np.log(NOG).diff()
NOG_var = NOG_var.dropna()
NOG_logf = NOG_logf.dropna()
NOG_rend = np.mean(NOG_logf)
NOG_std = NOG_logf.std()
print(NOG_rend)
print(NOG_std)

BANC_var = BANC.pct_change()
BANC_logf = np.log(BANC).diff()
BANC_var = BANC_var.dropna()
BANC_logf = BANC_logf.dropna()
BANC_rend = np.mean(BANC_logf)
BANC_std = BANC_logf.std()
print(BANC_rend)
print(BANC_std)

GM_var = GM.pct_change()
GM_logf = np.log(GM).diff()
GM_var = GM_var.dropna()
GM_logf = GM_logf.dropna()
GM_rend = np.mean(GM_logf)
GM_std = GM_logf.std()
print(GM_rend)
print(GM_std)

WNC_var = WNC.pct_change()
WNC_logf = np.log(WNC).diff()
WNC_var = WNC_var.dropna()
WNC_logf = WNC_logf.dropna()
WNC_rend = np.mean(WNC_logf)
WNC_std = WNC_logf.std()
print(WNC_rend)
print(WNC_std)

UAL_var = UAL.pct_change()
UAL_logf = np.log(UAL).diff()
UAL_var = UAL_var.dropna()
UAL_logf = UAL_logf.dropna()
UAL_rend = np.mean(UAL_logf)
UAL_std = UAL_logf.std()
print(UAL_rend)
print(UAL_std)

VTLE_var = VTLE.pct_change()
VTLE_logf = np.log(VTLE).diff()
VTLE_var = VTLE_var.dropna()
VTLE_logf = VTLE_logf.dropna()
VTLE_rend = np.mean(VTLE_logf)
VTLE_std = VTLE_logf.std()
print(VTLE_rend)
print(VTLE_std)

TSE_var = TSE.pct_change()
TSE_logf = np.log(TSE).diff()
TSE_var = TSE_var.dropna()
TSE_logf = TSE_logf.dropna()
TSE_rend = np.mean(TSE_logf)
TSE_std = TSE_logf.std()
print(TSE_rend)
print(TSE_std)

TPC_var = TPC.pct_change()
TPC_logf = np.log(TPC).diff()
TPC_var = TPC_var.dropna()
TPC_logf = TPC_logf.dropna()
TPC_rend = np.mean(TPC_logf)
TPC_std = TPC_logf.std()
print(TPC_rend)
print(TPC_std)

PDM_var = PDM.pct_change()
PDM_logf = np.log(PDM).diff()
PDM_var = PDM_var.dropna()
PDM_logf = PDM_logf.dropna()
PDM_rend = np.mean(PDM_logf)
PDM_std = PDM_logf.std()
print(PDM_rend)
print(PDM_std)

VVX_var = VVX.pct_change()
VVX_logf = np.log(VVX).diff()
VVX_var = VVX_var.dropna()
VVX_logf = VVX_logf.dropna()
VVX_rend = np.mean(VVX_logf)
VVX_std = VVX_logf.std()
print(VVX_rend)
print(VVX_std)

IMKTA_var = IMKTA.pct_change()
IMKTA_logf = np.log(IMKTA).diff()
IMKTA_var = IMKTA_var.dropna()
IMKTA_logf = IMKTA_logf.dropna()
IMKTA_rend = np.mean(IMKTA_logf)
IMKTA_std = IMKTA_logf.std()
print(IMKTA_rend)
print(IMKTA_std)

FNLC_var = FNLC.pct_change()
FNLC_logf = np.log(FNLC).diff()
FNLC_var = FNLC_var.dropna()
FNLC_logf = FNLC_logf.dropna()
FNLC_rend = np.mean(FNLC_logf)
FNLC_std = FNLC_logf.std()
print(FNLC_rend)
print(FNLC_std)

#UNIR ACTIVOS
CARTERA0 = pd.concat([GCO,NVDA,CXW,EAT,OSIS,KMT,DAN,EBF,SCS,MLKN,TILE,QUAD,CZNC,MRO,CAC,GILT,WAFD,GEF,CMC,CNOB,FULT,KBH,HVT,PEBO,FFBC,MPC,NMIH,CATY,MTG,RDN,DAL,F,NOG,BANC,GM,WNC,UAL,VTLE,TSE,TPC,PDM,VVX,IMKTA,FNLC],axis=1)
Rendimientos = pd.concat([GCO_logf,NVDA_logf,CXW_logf,EAT_logf,OSIS_logf,KMT_logf,DAN_logf,EBF_logf,SCS_logf,MLKN_logf,TILE_logf,QUAD_logf,CZNC_logf,MRO_logf,CAC_logf,GILT_logf,WAFD_logf,GEF_logf,CMC_logf,CNOB_logf,FULT_logf,KBH_logf,HVT_logf,PEBO_logf,FFBC_logf,MPC_logf,NMIH_logf,CATY_logf,MTG_logf,RDN_logf,DAL_logf,F_logf,NOG_logf,BANC_logf,GM_logf,WNC_logf,UAL_logf,VTLE_logf,TSE_logf,TPC_logf,PDM_logf,VVX_logf,IMKTA_logf,FNLC_logf],axis=1)

#GRAFICO DE COMPORTAMIENTO DE LOS ACTIVOS
import matplotlib.pyplot as plt
plt.figure(figsize=(12.2,4)) 
for i in CARTERA0.columns.values:
    plt.plot(CARTERA0[i], label=i)
plt.title("Price of the Stocks")
plt.xlabel("Date",fontsize=8)
plt.ylabel("Price",fontsize=8)
plt.legend(CARTERA0.columns.values, loc="upper left")
plt.show() 

#################PORTAFOLIO#######################
#Matriz de correlación 
Rendimientos1 = pd.concat([GCO_logf,NVDA_logf,CXW_logf,EAT_logf,OSIS_logf,KMT_logf,DAN_logf,EBF_logf,SCS_logf,MLKN_logf,TILE_logf,QUAD_logf,CZNC_logf,MRO_logf,CAC_logf,GILT_logf,WAFD_logf,GEF_logf,CMC_logf,CNOB_logf,FULT_logf,KBH_logf,HVT_logf,PEBO_logf,FFBC_logf,MPC_logf,NMIH_logf,CATY_logf,MTG_logf,RDN_logf,DAL_logf,F_logf,NOG_logf,BANC_logf,GM_logf,WNC_logf,UAL_logf,VTLE_logf,TSE_logf,TPC_logf,PDM_logf,VVX_logf,IMKTA_logf,FNLC_logf],axis=1)
Rendimientos1.corr()

#   Mapa De Calor 
import seaborn as sns
correlation_mat = Rendimientos1.corr()
plt.figure(figsize=(12.2,4.5))
sns.heatmap(correlation_mat, annot = True)
plt.title("Matriz de Correlación") 
plt.xlabel("Activos",fontsize=18)
plt.ylabel("Activos",fontsize=18)
plt.show()







#DESPUÉS DE ELEGIR A LAS VARIABLES, CREAMOS NUESTRA LISTA
#TSLA-FDX 
Rendimientos2 = pd.concat([TSLA_logf,FDX_logf],axis=1)
lista_stocks =  ["TSLA_rend","FDX_rend"]
print(lista_stocks)
num_stocks = len(lista_stocks) #Contará los activos
print(num_stocks) 

# Crear Variables
portafolioR = []
portafolioSD = []
portafolioPESOS = [] 

#CREAR PESOS
for x in range (5000):
    pesos = np.random.random(num_stocks)
    pesos /= np.sum(pesos)
    portafolioPESOS.append(pesos)
    portafolioR.append(np.dot(Rendimientos2.mean(),pesos))
    portafolioSD.append(np.sqrt(np.dot(pesos.T, np.dot(Rendimientos2.cov(), pesos))))
#PESOS
print(portafolioPESOS)
print(portafolioR)
print(portafolioSD)

#DICCIONARIO DE RENDIMIENTOS, RIESGOS Y PESOS (organiza)
diccionario = {"Rendimientos2":portafolioR, "Riesgo":portafolioSD}
for contador, ticker in enumerate(Rendimientos2.columns.tolist()):
    diccionario["Peso" + ticker] = [w[contador] for w in portafolioPESOS]
print(diccionario)
    
matrizportafolios = pd.DataFrame(diccionario) #ordenar en DATA FRAME
print(matrizportafolios) 

#PORTAFOLIO EFICIENTE: MÍNIMA VARIANZA, MENOR RIESGO
EFICIENTE = matrizportafolios.plot(x= "Riesgo", y="Rendimientos2", kind = "scatter") 
#ver rendimiento, riesgo y pesos
varianzaminima = matrizportafolios.iloc[matrizportafolios["Riesgo"].idxmin()]
plt.scatter(x = varianzaminima[1], y = varianzaminima[0], color = "red", marker = "*", s = 100)
print(varianzaminima)

#PORTAFOLIO ÓPTIMO O TANGENTE: MAXIMIZA LA RENTABILIDAD

rf = 0
matrizportafolios['Sharpe Ratio'] = (matrizportafolios['Rendimientos2'] - rf) / matrizportafolios['Riesgo']
indice_optimo = matrizportafolios['Sharpe Ratio'].idxmax()
#ver rendimiento, riesgo y pesos
OPTIMO = matrizportafolios.loc[indice_optimo]
matrizportafolios.plot(x = 'Riesgo', y = 'Rendimientos2', kind = 'scatter')
plt.scatter(x=varianzaminima[1], y=varianzaminima[0], color ='red', marker = '*', s=100)
plt.scatter(x=OPTIMO['Riesgo'], y=OPTIMO['Rendimientos2'], color='black', marker='*', s=100)

print(OPTIMO)

#CALCULAR RATIO SHARPE

RATIO_SHARPE_O = (OPTIMO["Rendimientos2"]-rf)/OPTIMO["Riesgo"]
#Cuando un Ratio de Sharpe es negativo indica que su rendimiento es menor al de la rentabilidad sin riesgo.
se toma el eficeinte

"El portafolio eficiente asume normalidad en los rendimientos; sin embargo, el portafolio"
 "optimo muestra que hay un riesgo que engloba la no normalidad de los rendmientos"
" óptimo el cual maximice la rentabilidad sujeta a una restricción de riesgo de pérdida y no solo a una medida de riesgo paramétrica como la desviación estándar."
#si se tiene dos portafolios optimos... se elige el que tiene menor ratio sharpe 

#TSLA-AAPL
Rendimientos3 = pd.concat([TSLA_logf,AAPL_logf],axis=1)
lista_stocks =  ["TSLA_rend","AAPL_rend"]
print(lista_stocks)
num_stocks = len(lista_stocks) #Contará los activos
print(num_stocks) 

# Crear Variables
portafolioR = []
portafolioSD = []
portafolioPESOS = [] 

#CREAR PESOS
for x in range (5000):
    pesos = np.random.random(num_stocks)
    pesos /= np.sum(pesos)
    portafolioPESOS.append(pesos)
    portafolioR.append(np.dot(Rendimientos3.mean(),pesos))
    portafolioSD.append(np.sqrt(np.dot(pesos.T, np.dot(Rendimientos3.cov(), pesos))))
#PESOS
print(portafolioPESOS)
print(portafolioR)
print(portafolioSD)

#DICCIONARIO DE RENDIMIENTOS, RIESGOS Y PESOS (organiza)
diccionario = {"Rendimientos3":portafolioR, "Riesgo":portafolioSD}
for contador, ticker in enumerate(Rendimientos3.columns.tolist()):
    diccionario["Peso" + ticker] = [w[contador] for w in portafolioPESOS]
print(diccionario)
    
matrizportafolios = pd.DataFrame(diccionario) #ordenar en DATA FRAME
print(matrizportafolios) 

#PORTAFOLIO EFICIENTE: MÍNIMA VARIANZA, MENOR RIESGO
EFICIENTE = matrizportafolios.plot(x= "Riesgo", y="Rendimientos3", kind = "scatter") 
#ver rendimiento, riesgo y pesos
varianzaminima = matrizportafolios.iloc[matrizportafolios["Riesgo"].idxmin()]
plt.scatter(x = varianzaminima[1], y = varianzaminima[0], color = "red", marker = "*", s = 100)
print(varianzaminima)

#PORTAFOLIO ÓPTIMO O TANGENTE: MAXIMIZA LA RENTABILIDAD

rf = 0
matrizportafolios['Sharpe Ratio'] = (matrizportafolios['Rendimientos3'] - rf) / matrizportafolios['Riesgo']
indice_optimo = matrizportafolios['Sharpe Ratio'].idxmax()
#ver rendimiento, riesgo y pesos
OPTIMO = matrizportafolios.loc[indice_optimo]
matrizportafolios.plot(x = 'Riesgo', y = 'Rendimientos3', kind = 'scatter')
plt.scatter(x=varianzaminima[1], y=varianzaminima[0], color ='red', marker = '*', s=100)
plt.scatter(x=OPTIMO['Riesgo'], y=OPTIMO['Rendimientos3'], color='black', marker='*', s=100)

print(OPTIMO)

#FDX-AAPL
Rendimientos4 = pd.concat([FDX_logf,AAPL_logf],axis=1)
lista_stocks =  ["FDX_rend","AAPL_rend"]
print(lista_stocks)
num_stocks = len(lista_stocks) #Contará los activos
print(num_stocks) 

# Crear Variables
portafolioR = []
portafolioSD = []
portafolioPESOS = [] 

#CREAR PESOS
for x in range (5000):
    pesos = np.random.random(num_stocks)
    pesos /= np.sum(pesos)
    portafolioPESOS.append(pesos)
    portafolioR.append(np.dot(Rendimientos3.mean(),pesos))
    portafolioSD.append(np.sqrt(np.dot(pesos.T, np.dot(Rendimientos3.cov(), pesos))))
#PESOS
print(portafolioPESOS)
print(portafolioR)
print(portafolioSD)

#DICCIONARIO DE RENDIMIENTOS, RIESGOS Y PESOS (organiza)
diccionario = {"Rendimientos4":portafolioR, "Riesgo":portafolioSD}
for contador, ticker in enumerate(Rendimientos4.columns.tolist()):
    diccionario["Peso" + ticker] = [w[contador] for w in portafolioPESOS]
print(diccionario)
    
matrizportafolios = pd.DataFrame(diccionario) #ordenar en DATA FRAME
print(matrizportafolios) 

#PORTAFOLIO EFICIENTE: MÍNIMA VARIANZA, MENOR RIESGO
EFICIENTE = matrizportafolios.plot(x= "Riesgo", y="Rendimientos4", kind = "scatter") 
#ver rendimiento, riesgo y pesos
varianzaminima = matrizportafolios.iloc[matrizportafolios["Riesgo"].idxmin()]
plt.scatter(x = varianzaminima[1], y = varianzaminima[0], color = "red", marker = "*", s = 100)
print(varianzaminima)

#PORTAFOLIO ÓPTIMO O TANGENTE: MAXIMIZA LA RENTABILIDAD

rf = 0
matrizportafolios['Sharpe Ratio'] = (matrizportafolios['Rendimientos4'] - rf) / matrizportafolios['Riesgo']
indice_optimo = matrizportafolios['Sharpe Ratio'].idxmax()
#ver rendimiento, riesgo y pesos
OPTIMO = matrizportafolios.loc[indice_optimo]
matrizportafolios.plot(x = 'Riesgo', y = 'Rendimientos4', kind = 'scatter')
plt.scatter(x=varianzaminima[1], y=varianzaminima[0], color ='red', marker = '*', s=100)
plt.scatter(x=OPTIMO['Riesgo'], y=OPTIMO['Rendimientos4'], color='black', marker='*', s=100)

print(OPTIMO)