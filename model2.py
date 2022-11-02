# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 17:48:45 2021

@author: 41618
"""
import sys
sys.path.append(r'C:\Users\41618\Desktop\Shenyi Data\model\Shen_Yi_project')
#import technical_indicator as tec
import pre
import os
import pandas as pd
import numpy as np
import math
import calendar
import sys
import matplotlib.pyplot as plt
import datetime
import copy
#import model
plt.style.use('fivethirtyeight')


global month, year
month = ['01', '02', '03', '04', '05', '06',
         '07', '08', '09', '10', '11', '12']
year = ['2010', '2011', '2012', '2013', '2014', '2015',
        '2016', '2017', '2018', '2019', '2020', '2021']


def read_data():

    global monthly_data, annually_data, quarterly_data, info, datalist
    path = r'C:\Users\41618\Desktop\行业景气度\财务指标'
    datalist = os.listdir(path)
    a_cnt = 0 # 年度数据的数量
    q_cnt = 0 # 季度数据的数量
    m_cnt = 0 # 月度数据的数量
    for i in datalist:
        if i == 'info.csv':
            continue
        tmp = pd.read_csv(path+'\\'+i)
        co = ['Code']
        print(i);
        for j in range(1, tmp.shape[1]):
            print(tmp.columns[j])
            t = pd.to_datetime(tmp.columns[j]) ## 调整日期格式
            co.append(i[:-4]+'_'+str(t)[:10])
            try: ## 去除数字中的逗号
                tmp[tmp.columns[j]] = tmp[tmp.columns[j]].apply(
                    lambda x: x.replace(',', '')) 
                tmp[tmp.columns[j]] = tmp[tmp.columns[j]].astype('float')
            except:
                tmp[tmp.columns[j]] = tmp[tmp.columns[j]].astype('float')
        tmp.columns = co ##对列名重命名，使列名等于类似asset_quarterly_2021-06-30
        tmp.replace(0,np.nan,inplace=True);
        if 'annually' in i:
            if a_cnt == 0:
                annually_data = tmp
            else:
                annually_data = pd.merge(
                    left=annually_data, right=tmp, on='Code')
            a_cnt = a_cnt+1
        if 'monthly' in i:
            if m_cnt == 0:
                monthly_data = tmp
            else:
                monthly_data = pd.merge(
                    left=monthly_data, right=tmp, on='Code')
            m_cnt = m_cnt+1
        if 'quarter' in i:
            if q_cnt == 0:
                quarterly_data = tmp
            else:
                quarterly_data = pd.merge(
                    left=quarterly_data, right=tmp, on='Code')
            q_cnt = q_cnt+1
    info = pd.read_csv(path+'\\'+'info.csv')
    monthly_data = pd.merge(left=monthly_data, right=info, on='Code')
    quarterly_data = pd.merge(left=quarterly_data, right=info, on='Code')
    annually_data = pd.merge(left=annually_data, right=info, on='Code')

## type1 npr,ocfi,cloca,loa
def type1(merged_df,co1,co2,df,feature):
    co1 = 'net_profit_ttm_monthly'
    co2 = 'operating_income_ttm_monthly'
    tmp = pd.DataFrame(df['Code'])
    for i in range(1, len(year)):
        for j in range(len(month)):
            if i == 11 and j == 7:
                break;
            tmp[cn(feature, i, j)] = df[cn(co1, i, j)]/df[cn(co2, i, j)]
    merged_df = pd.merge(left=merged_df, right=tmp, on='Code')
    return merged_df

######################################################
def quick_ratio(df):
    co2 = 'current_liability_quarter'
    co1 = 'current_asset_quarter'
    co3 = 'net_inventory_quarter'
    qr = pd.DataFrame(df['Code'])
    for i in range(1, len(year)):
        for j in range(len(month)):
            if i == 11 and j == 7:
                break
            tmp1 = ((df[cn(co1, i, j)]-df[cn(co3, i, j)])/df[cn(co2, i, j)])
            qr['qr_'+year[i]+'-'+month[j]] = tmp1
    return qr

#########################



#######################
def pecut_zhengti(df):
    co1 = 'net_profit_ne_quarter'
    co2 = 'value_monthly'
    pecut = pd.DataFrame(df['Code'])
    for i in range(1, len(year)):
        for j in range(len(month)):
            if i == 11 and j == 7:
                break
            tmp1 = df[cn(co1, i, j)]
            tmp2 = df[cn(co2, i, j)]
            # tmp2=(df[cn(co1,i-1,j)]/df[cn(co2,i-1,j)])
            pecut[cn('pecut_numerator', i, j)] = tmp1
            pecut[cn('pecut_denominator', i, j)] = tmp2
    return pecut

#type 2 tr, huanshoulv,this month_re,rtr,npm,re12,t,tradevolume
def type2(merged_df,co1,df,feature):
    tmp = pd.DataFrame(df['Code'])
    for i in range(1, len(year)):
        for j in range(len(month)):
            if i == 11 and j == 7:
                break
            tmp[cn(feature, i, j)] = df[cn(co1, i, j)]
            merged_df = pd.merge(left=merged_df, right=tmp, on='Code')
    return merged_df;


###########type 3 npi,roei,oii
def type3(merged_df,co1,co2,df, df1,feature):
    co1 = 'est_net_profit_monthly'
    co2 = 'net_profit_p_ttm_annually'
    fe_nu=feature+'_numerator';
    fe_de=feature+'_denominator';
    
    tmp = pd.DataFrame(df['Code'])
    for i in range(3, len(year)):
        if i != len(year)-1:
            y1 = (df1[co2+"_"+year[i]]+df1[co2+"_" +
                  year[i-1]]+df1[co2+"_"+year[i-2]])/3
        y2 = (df1[co2+"_"+year[i-3]]+df1[co2+"_" +
              year[i-1]]+df1[co2+"_"+year[i-2]])/3
        for j in range(len(month)):
            if i == 11 and j == 7:
                break
            if j > 9:
                tmp[cn('npi_numerator', i, j)] = df[cn(co1, i, j)]-y1
                tmp[cn('npi_denominator', i, j)] = y1.apply(
                    lambda x: np.abs(x))
            else:
                tmp[cn('npi_numerator', i, j)] = df[cn(co1, i, j)]-y2
                tmp[cn('npi_denominator', i, j)] = y2.apply(
                    lambda x: np.abs(x))
    merged_df = pd.merge(left=merged_df, right=qr, on='Code')
    return merged_df
def net_profit_increment(df, df1):
    co1 = 'est_net_profit_monthly'
    co2 = 'net_profit_p_ttm_annually'
    npi = pd.DataFrame(df['Code'])
    for i in range(3, len(year)):
        if i != len(year)-1:
            y1 = (df1[co2+"_"+year[i]]+df1[co2+"_" +
                  year[i-1]]+df1[co2+"_"+year[i-2]])/3
        y2 = (df1[co2+"_"+year[i-3]]+df1[co2+"_" +
              year[i-1]]+df1[co2+"_"+year[i-2]])/3
        for j in range(len(month)):
            if i == 11 and j == 7:
                break
            if j > 9:
                npi[cn('npi_numerator', i, j)] = df[cn(co1, i, j)]-y1
                npi[cn('npi_denominator', i, j)] = y1.apply(
                    lambda x: np.abs(x))
            else:
                npi[cn('npi_numerator', i, j)] = df[cn(co1, i, j)]-y2
                npi[cn('npi_denominator', i, j)] = y2.apply(
                    lambda x: np.abs(x))
    return npi
def roe_increment(df, df1):
    co1 = 'est_roe_monthly'
    co2 = 'roe_ttm_annually'
    roei = pd.DataFrame(df['Code'])
    for i in range(3, len(year)):
        if i != len(year)-1:
            y1 = (df1[co2+"_"+year[i]]+df1[co2+"_" +
                  year[i-1]]+df1[co2+"_"+year[i-2]])/3
        y2 = (df1[co2+"_"+year[i-3]]+df1[co2+"_" +
              year[i-1]]+df1[co2+"_"+year[i-2]])/3
        for j in range(len(month)):
            if i == 11 and j == 7:
                break
            if j > 9:
                roei[cn('roei_numerator', i, j)] = df[cn(co1, i, j)]-y1
                roei[cn('roei_denominator', i, j)] = y1.apply(
                    lambda x: np.abs(x))
            else:
                roei[cn('roei_numerator', i, j)] = df[cn(co1, i, j)]-y2
                roei[cn('roei_denominator', i, j)] = y2.apply(
                    lambda x: np.abs(x))
    return roei


def operating_income_increment(df, df1):
    co1 = 'est_operating_income_ttm_monthly'
    co2 = 'operating_income_ttm_annually'
    oii = pd.DataFrame(df['Code'])
    for i in range(3, len(year)):
        if i != len(year)-1:
            y1 = (df1[co2+"_"+year[i]]+df1[co2+"_" +
                  year[i-1]]+df1[co2+"_"+year[i-2]])/3
        y2 = (df1[co2+"_"+year[i-3]]+df1[co2+"_" +
              year[i-1]]+df1[co2+"_"+year[i-2]])/3
        for j in range(len(month)):
            if i == 11 and j == 7:
                break
            if j > 9:
                oii[cn('oii_numerator', i, j)] = df[cn(co1, i, j)]-y1
                oii[cn('oii_denominator', i, j)] = y1.apply(
                    lambda x: np.abs(x))
            else:
                oii[cn('oii_numerator', i, j)] = df[cn(co1, i, j)]-y2
                oii[cn('oii_denominator', i, j)] = y2.apply(
                    lambda x: np.abs(x))
    return oii

################################################
##
def inventory_turnover_ratio(df):  # 一半》0
    co = 'inventory_turnover_quarter'
    itr = pd.DataFrame(df['Code'])
    for i in range(1, len(year)):
        for j in range(len(month)):
            if i == 11 and j == 7:
                break
            tmp1 = df[co+'_'+year[i]+'-'+month[j]]
            itr['itr_'+year[i]+'-'+month[j]] = tmp1
    return itr

def cal_roettm_zhengti(df):
    co1 = 'net_profit_p_ttm_monthly'
    co2 = 'equity_p_quarter'
    roettm = pd.DataFrame(df['Code'])
    for i in range(1, len(year)):
        for j in range(len(month)):
            if i == 11 and j == 7:
                break
            tmp = df[cn(co1, i, j)]
            if j == 3:
                tmp1 = (df[cn(co2, i-1, (j+4) % 12)]+df[cn(co2, i, j)])/2
            elif j == 6:
                tmp1 = (df[cn(co2, i-1, j+2)]+df[cn(co2, i, j)])/2
            else:
                tmp1 = (df[cn(co2, i-1+((j+3)//12), (j+3) % 12)] +
                        df[cn(co2, i, j)])/2
            roettm[cn('roe_ttm_numerator', i, j)] = tmp
            roettm[cn('roe_ttm_denominator', i, j)] = tmp1
            # roettm[cn(co1,i,j)]=tmp;
            # roettm[cn(co2,i,j)]=tmp1;
    return roettm





def quarter_month(df):
    tmp = df[df.columns[0]]
    tmp = pd.DataFrame(tmp)
    for i in df.columns:

        if '03-31' in i:
            tmp[i[:-5]+'04'] = df.loc[:, i]
            tmp[i[:-5]+'05'] = df.loc[:, i]
            tmp[i[:-5]+'06'] = df.loc[:, i]
            tmp[i[:-5]+'07'] = df.loc[:, i]
        if '06-30' in i:

            tmp[i[:-5]+'08'] = df.loc[:, i]
            tmp[i[:-5]+'09'] = df.loc[:, i]
        if '09-30' in i:
            tmp[i[:-5]+'10'] = df.loc[:, i]
            tmp[i[:-5]+'11'] = df.loc[:, i]
            tmp[i[:-5]+'12'] = df.loc[:, i]
        if '12-31' in i:

            tmp[i[:-8]+str(int(i[-8:-6])+1)+'-01'] = df.loc[:, i]
            tmp[i[:-8]+str(int(i[-8:-6])+1)+'-02'] = df.loc[:, i]
            tmp[i[:-8]+str(int(i[-8:-6])+1)+'-03'] = df.loc[:, i]
    return tmp








def cn(string, i, j):
    return string+'_'+year[i]+'-'+month[j]


def getcode(info, t, first_level_ind, sec_level_ind):
    info['上市日期'] = pd.to_datetime(info['上市日期'])
    first = info.copy()
    second = info.copy()
    first.index = info['一级行业']
    second.index = info['二级行业']

    first = first.loc[first_level_ind].drop(columns='一级行业').reset_index()
    first['行业'] = first['一级行业']
    second = second.loc[sec_level_ind].drop(columns='二级行业').reset_index()
    second['行业'] = second['二级行业']
    first = first.append(second, ignore_index=True)
    first['signal'] = first['上市日期'].apply(
        lambda x: 0 if (x-t).days < 365 else 1)
    first = first[first['signal'] == 0]
    first = first.reset_index().drop(columns=['index', 'signal'])
    return first


def cal_feature(df, string):
    global me
    co = list(df.columns)
    for i in range(len(co)):
        co[i] = co[i].replace(string, '')
    df.columns = co
    s = df.groupby(by='行业').sum()
    me = df.groupby(by='行业').median()

    ####整体
    d=df[['roe_ttm_numerator','roe_ttm_denominator','行业']].copy();
    d.dropna(inplace=True);
    tmp=d.groupby(by='行业').sum();
    # 整体
    s['roe_ttm'] = tmp['roe_ttm_numerator']/tmp['roe_ttm_denominator']
    s['npr'] = me['npr']
    s['ocfi'] = me['ocfi']
    d=df[['oii_numerator','oii_denominator','行业']].copy();
    d.dropna(inplace=True);
    tmp=d.groupby(by='行业').sum();
    s['oii'] = tmp['oii_numerator']/tmp['oii_denominator']
    d=df[['roei_numerator','roei_denominator','行业']].copy();
    d.dropna(inplace=True);
    tmp=d.groupby(by='行业').sum();
    s['roei'] = tmp['roei_numerator']/tmp['roei_denominator']
    d=df[['npi_numerator','npi_denominator','行业']].copy();
    d.dropna(inplace=True);
    tmp=d.groupby(by='行业').sum();
    s['npi'] = tmp['npi_numerator']/tmp['npi_denominator']
    d=df[['pecut_numerator','pecut_denominator','行业']].copy();
    d.dropna(inplace=True);
    tmp=d.groupby(by='行业').sum();
    s['pecut'] = tmp['pecut_numerator']/tmp['pecut_denominator']
    s['loa'] = me['loa']
    s['t']=me['t'];
    s['re12']=me['re12']
    s['huanshoulv']=me['huanshoulv'];
    s['this_month_re']=me['this_month_re'];
    s['tradevolume']=me['tradevolume'];
    s['cloca'] = me['cloca']
    s['qr'] = me['qr']
    s['itr'] = me['itr']
    s['np'] = me['np']
    s = s.loc[:, ['roe_ttm', 'npr', 'ocfi', 'oii', 'roei', 'npi',
                  'loa', 'cloca', 'rtr', 'qr', 'itr', 'tr', 'np', 'pecut','t','re12','huanshoulv','this_month_re','tradevolume']]
    co = list(s.columns)
    for i in range(len(co)):
        if co[i] == 'Code':
            continue
        else:
            co[i] = co[i]+string
    s.columns = co
    return s


def huanbi(df, co, y, m):
    if m == 6:
        se = (df[cn(co, y, m)]-df[cn(co, y, m-4)])/df[cn(co, y, m-4)]
    elif m == 9:
        se = (df[cn(co, y, m)]-df[cn(co, y, m-2)])/df[cn(co, y, m-2)]
    else:
        se = (df[cn(co, y, m)]-df[cn(co, y-1+(m+9)//12, (m+9)%12)]) / \
            df[cn(co, y-1+(m+9)//12,(m+9)%12)]
    return se


def tongbi(df, co, y, m):
    se = (df[cn(co, y, m)]-df[cn(co, y-1, m)])/df[cn(co, y-1, m)]
    return se


if __name__ == '__main__':
    read_data()
    ###
    p = r'C:\Users\41618\Desktop\行业景气度' 
    pblf = pd.read_csv(os.path.join(p, 'pblf_monthly.csv'))
    pettm = pd.read_csv(os.path.join(p, 'pettm_monthly.csv'))
    co = ['行业']
    for i in range(1, pblf.shape[1]):
        co.append('pblf_'+pblf.columns[i][:7])
    pblf.columns = co
    co = ['行业']
    for i in range(1, pettm.shape[1]):
        co.append('pettm_'+pettm.columns[i][:7])
    pettm.columns = co
    pblf.index = pettm['行业']
    pblf.drop(columns='行业', inplace=True)
    pettm.index = pettm['行业']
    pettm.drop(columns='行业', inplace=True)
    ##
    
    ##只关注这两天
    first_level_ind = ['钢铁', '纺织服装', '轻工制造', '建材',
                        '传媒', '国防军工', '商贸零售', '银行', '交通运输', '农林牧渔']
    sec_level_ind = ['其他化学制品Ⅱ', '生物医药Ⅱ', '石油化工', '通用设备',
                      '建筑施工', '计算机软件', '电源设备', '煤炭开采洗选', '证券Ⅱ', '白色家电Ⅱ', '其他医药医疗', '乘用车Ⅱ',
                      '光学光电', '旅游及休闲', '电气设备', '汽车零部件Ⅱ',
                      '新能源动力系统', '发电及电网', '专用机械',
                      '化学制药', '中药生产', '保险Ⅱ', '稀有金属',
                      '半导体', '通信设备制造', '食品', '消费电子', '酒类', '房地产开发和运营']
    ###
    agg_f_info = info.groupby(by='一级行业').count()
    a = agg_f_info.loc[first_level_ind]
    agg_s_info = info.groupby(by='二级行业').count()
    b = agg_s_info.loc[sec_level_ind]
    qmdata = quarter_month(quarterly_data)## 季度数据转成月度数据
    ### 统一列名
    co = ['Code']
    for i in monthly_data.columns:
        if i == 'Code':
            continue
        co.append(i[:-3])
    monthly_data.columns = co
    ## 
    monthly_data = pd.merge(left=monthly_data, right=qmdata, on='Code')
    ## 把final
    sys.exit();
    final_df = info




## type1 npr,ocfi,cloca,loa
#type 2 tr, huanshoulv,this month_re,rtr,npm,re12,t,tradevolume
###########type 3 npi,roei,oii
    # preprocessing annually
    co = ['Code']
    for i in range(1, annually_data.shape[1]):
        co.append(annually_data.columns[i][:-6])
    annually_data.columns = co

    npi = net_profit_increment(monthly_data, annually_data)
    final_df = pd.merge(left=final_df, right=npi, on='Code')
    oii = operating_income_increment(monthly_data, annually_data)
    final_df = pd.merge(left=final_df, right=oii, on='Code')
    roei = roe_increment(monthly_data, annually_data)
    final_df = pd.merge(left=final_df, right=roei, on='Code')

    final_df = type1(final_df,'net_profit_ttm_monthly','operating_income_ttm_monthly',monthly_data,'npr');
    final_df = type1(final_df,'operating_cf_ttm_monthly','operating_income_ttm_monthly',monthly_data,'ocfi');
    final_df = type1(final_df,'current_asset_quarte','current_liability_quarterr',monthly_data,'cloca');
    final_df = type1(final_df,'liability_quarter','asset_quarterly',monthly_data,'loa');
    final_df = type1(final_df,'net_profit_ttm_monthly',monthly_data,'npm');
    final_df = type1(final_df,'huanshoulv_monthly',monthly_data,'huanshoulv');
    final_df = type1(final_df,'this_month_re_monthly',monthly_data,'this_month_re');
    final_df = type1(final_df,'receivables_turnover_ratio_quarter',monthly_data,'rtr');
    final_df = type1(final_df,'turnover_rate_quarter',monthly_data,'re12');
    final_df = type1(final_df,'t_monthly',monthly_data,'t');
    final_df = type1(final_df,'tradevolume_monthly',monthly_data,'nptradevolumem');
    

    qr = quick_ratio(monthly_data)
    final_df = pd.merge(left=final_df, right=qr, on='Code')

    roettm = cal_roettm_zhengti(monthly_data)
    final_df = pd.merge(left=final_df, right=roettm, on='Code')

    itr = inventory_turnover_ratio(monthly_data)
    final_df = pd.merge(left=final_df, right=itr, on='Code')

    
    pecut = pecut_zhengti(monthly_data)
    final_df = pd.merge(left=final_df, right=pecut, on='Code')
    




    cnt = 0
    for i in range(2015, 2022):
        for j in range(1, 13):
            co = ['Code']
            if i == 2021 and j == 8:
                break
            for k in final_df.columns:
                if cn('', i-2010, j-1) in k:
                    co.append(k)
            t = datetime.datetime(i+1, j, 28)
            tdf = getcode(info, t, first_level_ind, sec_level_ind)
            tmp = pd.merge(left=final_df[co], right=tdf[[
                           'Code', '行业']], on='Code', how='right')
            for k in tmp.columns:
                if k == 'Code' or k == '行业':
                    continue
                elif 'numerator' in k:
                    continue
                elif 'denominator' in k:
                    continue
                else:
                    tmp[k] = pre.checkdata(tmp[k], k, price=False)
            # score=cal_score(tmp,cn('',i-2010,j-1));
            tmp1 = cal_feature(tmp, cn('', i-2010, j-1))
            if cnt == 0:
                feature_data = tmp1
                cnt = 1
            else:
                feature_data = pd.merge(
                    left=feature_data, right=tmp1, left_index=True, right_index=True)

    final_df = pd.DataFrame()
    for i in range(2016, 2022):
        for j in range(1, 13):
            if i == 2021 and j == 8:
                break
            final_df[cn('pettm', i-2010, j-1)
                     ] = pettm[cn('pettm', i-2010, j-1)]
            final_df[cn('pblf', i-2010, j-1)] = pblf[cn('pblf', i-2010, j-1)]
            final_df[cn('roe_ttm', i-2010, j-1)
                     ] = huanbi(feature_data, 'roe_ttm', i-2010, j-1)
            final_df[cn('npr', i-2010, j-1)] = huanbi(feature_data,
                                                      'npr', i-2010, j-1)
            final_df[cn('ocfi', i-2010, j-1)
                     ] = huanbi(feature_data, 'ocfi', i-2010, j-1)
            final_df[cn('loa', i-2010, j-1)] = tongbi(feature_data,
                                                      'loa', i-2010, j-1)

            final_df[cn('cloca', i-2010, j-1)
                     ] = tongbi(feature_data, 'cloca', i-2010, j-1)

            final_df[cn('qr', i-2010, j-1)] = tongbi(feature_data,
                                                     'qr', i-2010, j-1)
            final_df[cn('np', i-2010, j-1)] = tongbi(feature_data,
                                                     'np', i-2010, j-1)
            final_df[cn('peg', i-2010, j-1)] = final_df[cn('pettm',
                                                           i-2010, j-1)]/final_df[cn('np', i-2010, j-1)]
            final_df[cn('itr', i-2010, j-1)] = huanbi(feature_data,
                                                      'itr', i-2010, j-1)
            final_df[cn('tr', i-2010, j-1)] = huanbi(feature_data,
                                                     'tr', i-2010, j-1)
            final_df[cn('rtr', i-2010, j-1)] = huanbi(feature_data,
                                                      'rtr', i-2010, j-1)
            final_df[cn('npi_huanbi', i-2010, j-1)
                     ] = huanbi(feature_data, 'npi', i-2010, j-1)
            final_df[cn('oii_huanbi', i-2010, j-1)
                     ] = huanbi(feature_data, 'oii', i-2010, j-1)
            final_df[cn('roei_huanbi', i-2010, j-1)
                     ] = huanbi(feature_data, 'roei', i-2010, j-1)
            final_df[cn('npi', i-2010, j-1)
                     ] = feature_data[cn('npi', i-2010, j-1)]
            final_df[cn('oii', i-2010, j-1)
                     ] = feature_data[cn('oii', i-2010, j-1)]
            final_df[cn('roei', i-2010, j-1)
                     ] = feature_data[cn('roei', i-2010, j-1)]
            final_df[cn('pecut', i-2010, j-1)
                     ] = feature_data[cn('pecut', i-2010, j-1)]
            final_df[cn('t', i-2010, j-1)]=feature_data[cn('t', i-2010, j-1)]
            final_df[cn('re12', i-2010, j-1)]=feature_data[cn('re12', i-2010, j-1)]
            final_df[cn('this_month_re', i-2010, j-1)]=feature_data[cn('this_month_re', i-2010, j-1)]
            final_df[cn('tradevolume', i-2010, j-1)]=feature_data[cn('tradevolume', i-2010, j-1)]
            final_df[cn('huanshoulv', i-2010, j-1)]=feature_data[cn('huanshoulv', i-2010, j-1)]
    for i in range(2016, 2022):
        for j in range(1, 13):
            if i == 2021 and j == 8:
                break
            if i ==2016 and j>=1 and j<=3:
                continue;
            final_df[cn('loa_huanbi', i-2010, j-1)]=huanbi(final_df,'loa',i-2010,j-1)
            final_df[cn('cloca_huanbi', i-2010, j-1)]=huanbi(final_df,'cloca',i-2010,j-1)
            final_df[cn('qr_huanbi', i-2010, j-1)]=huanbi(final_df,'qr',i-2010,j-1)