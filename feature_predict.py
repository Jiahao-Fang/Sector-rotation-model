
from base import sheet,featuredata,score
import os
import pandas as pd
import sys
import time
def read_data():

    path = './data'
    datalist = os.listdir(path)
    for filename in datalist:
        for i,ch in enumerate(filename):
            if ch=='_':
                tmp=i
        globals()[filename[:tmp]]=sheet(pd.read_csv(path+'\\'+filename),filename[:tmp],filename[tmp+1:-4])

    
def cal_feature():
    fea_df.feature_generation_1(net_profit_ttm,operating_income_ttm,'npr')  ##
    fea_df.feature_generation_1(operating_cf_ttm,operating_income_ttm,'ocfi') ##
    fea_df.feature_generation_1(current_liability,current_asset,'cloca')##
    fea_df.feature_generation_1(liability,asset,'loa')##
    fea_df.feature_generation_2(huanshoulv,'huanshoulv')
    fea_df.feature_generation_2(re,'re')
    fea_df.feature_generation_2(tradevolume,'tradevolume')
    fea_df.feature_generation_2(t,'t')
    fea_df.feature_generation_2(re12,'re12')
    fea_df.feature_generation_2(net_profit_ttm,'npm')
    fea_df.feature_generation_2(turnover_rate,'trq')##
    fea_df.feature_generation_2(tradevolume,'tv')
    fea_df.feature_generation_2(receivables_turnover_ratio,'rtr')#
    fea_df.feature_generation_2(inventory_turnover,'itr')#
    fea_df.feature_generation_3(est_net_profit,net_profit_p_ttm,'npi')##
    fea_df.feature_generation_3(est_operating_income_ttm,operating_income_ttm,'oii')##
    fea_df.feature_generation_3(est_roe,roe_ttm,'roei')##
    fea_df.quick_ratio(current_asset,current_liability,net_inventory)##
    fea_df.cal_roettm_zhengti(net_profit_p_ttm,equity_p)##
    fea_df.pecut(net_profit_ne,value)
    fea_df.sector_filtering()
    print(fea_df.feature_df['行业'])
read_data()
info=pd.read_csv('./info.csv')
fea_df=featuredata(info)
cal_feature()

#$sc= score(info)
                     