import pandas as pd
import numpy as np
month = ['01', '02', '03', '04', '05', '06',
         '07', '08', '09', '10', '11', '12']
year = [ '2015','2016', '2017', '2018', '2019', '2020', '2021']
def cn(string, i, j):
    return string+'_'+year[i]+'-'+month[j]
class sheet:
    def __init__(self,df,filename,type):
        self.data=df
        self.name=filename
        self.type=type
        self.to_month_date()
        self.del_comma()
        self.quarter_to_month()
    def to_month_date(self):
        self.data.drop(columns='Code',inplace=True)
        self.data.columns=self.data.columns.to_series().apply(lambda x:pd.to_datetime(x))
        self.data.columns=self.data.columns.to_series().apply(lambda x:self.name+'_'+str(x)[:7])
    def del_comma(self):
        try:
            self.data=self.data.applymap(lambda x: x.replace(',', ''))
            self.data= self.data.astype('float')
        except:
            self.data= self.data.astype('float')
    def quarter_to_month(self):
        if self.type=='quarter':
            tmp=[]
            print(self.name)
            s=[['04','05','06','07'],['08','09'],['10','11','12']]
            t=['03','06','09']
            for co_name in self.data.columns:
                for i in range(3):
                    if t[i] in co_name:
                        for mo in s[i]:
                            tmp.append(pd.Series(self.data.loc[:, co_name],name=co_name[:-2]+mo))
                
                if '12' in co_name:
                    s4=['-01','-02','-03']
                    for mo in s4:
                        tmp.append(pd.Series(self.data.loc[:, co_name],name=co_name[:-5]+str(int(co_name[-5:-3])+1)+mo))
            self.data=pd.concat(tmp,axis=1)
class featuredata():
    def __init__(self,df):
        self.feature_df=df
    def feature_generation_1(self,sheet1,sheet2,feature_name):
        series_list=[]
        for i in range(1, len(year)):
            for j in range(len(month)):
                if i == 6 and j == 6:
                    break
                series_list.append(pd.Series(sheet1.data[cn(sheet1.name, i, j)],
                name=cn(feature_name'_numerator', i, j)))
                series_list.append(pd.Series(sheet2.data[cn(sheet2.name, i, j)],
                name=cn(feature_name+'_denominator', i, j)))
        self.feature_df=pd.concat([self.feature_df,*series_list],axis=1)
        
    def feature_generation_2(self,sheet,feature_name):
        series_list=[]
        for i in range(1, len(year)):
            for j in range(len(month)):
                if i == 6 and j == 6:
                    break
                series_list.append(pd.Series(sheet.data[cn(sheet.name, i, j)],
                name=cn(feature_name, i, j)))
        self.feature_df=pd.concat([self.feature_df,*series_list],axis=1)
    def feature_generation_3(self,sheet1,sheet2,feature_name):
        series_list=[]
        for i in range(3, len(year)):
            if i != len(year)-1:
                y1 = (sheet2.data[cn(sheet2.name, i, 11)]+sheet2.data[cn(sheet2.name, i-1, 11)]+
                sheet2.data[cn(sheet2.name, i-2, 11)])/3
            y2 = (sheet2.data[cn(sheet2.name, i-3, 11)]+sheet2.data[cn(sheet2.name, i-2, 11)]+
            sheet2.data[cn(sheet2.name, i-1, 11)])/3
            for j in range(len(month)):
                if i == 6 and j == 6:
                    break
                if j > 9:
                    ave_y=y1
                else:
                    ave_y=y2
                series_list.append(pd.Series(sheet1.data[cn(sheet1.name, i, j)]-ave_y,
                name=cn(feature_name+'_numerator', i, j)))
                series_list.append(pd.Series(ave_y.apply(lambda x: np.abs(x)),
                name=cn(feature_name+'_denominator', i, j)))
        self.feature_df=pd.concat([self.feature_df,*series_list],axis=1)
    def feature_generation_4()
    def quick_ratio(self,sheet1,sheet2,sheet3):
        series_list=[]
        for i in range(1, len(year)):
            for j in range(len(month)):
                if i == 6 and j == 6:
                    break
                series_list.append(pd.Series(sheet1.data[cn(sheet1.name,i,j)]-\
                    sheet3.data[cn(sheet3.name,i,j)],name=cn('qr_numerator',i,j)))
                series_list.append(pd.Series(sheet2.data[cn(sheet2.name,i,j)],name=cn('qr_denominator',i,j)))
                # series_list.append(pd.Series(((sheet1.data[cn(sheet1.name, i, j)]-\
                #     sheet3.data[cn(sheet3.name, i, j)])/sheet2.data[cn(sheet2.name, i, j)]),name=cn('qr',i,j)))
        self.feature_df=pd.concat([self.feature_df,*series_list],axis=1)
    def pecut(self,sheet1,sheet2):
        series_list=[]
        for i in range(1, len(year)):
            for j in range(len(month)):
                if i == 6 and j == 6:
                    break
                series_list.append(pd.Series(sheet1.data[cn(sheet1.name, i, j)],
                name=cn('pecut_numerator', i, j)))
                series_list.append(pd.Series(sheet2.data[cn(sheet2.name, i, j)],
                name=cn('pecut_denominator', i, j)))
        self.feature_df=pd.concat([self.feature_df,*series_list],axis=1)
    def cal_roettm_zhengti(self,sheet1,sheet2):
        series_list=[]
        for i in range(1, len(year)):
            for j in range(len(month)):
                if i == 6 and j == 6:
                    break
                series_list.append(pd.Series(sheet1.data[cn(sheet1.name, i, j)],name=cn('roe_ttm_numerator', i, j)))
                if j == 3:
                    series_list.append(pd.Series((sheet2.data[cn(sheet2.name, i-1, (j+4) % 12)]+\
                        sheet2.data[cn(sheet2.name, i, j)])/2,name=cn('roe_ttm_denominator', i, j)))
                elif j == 6:
                    series_list.append(pd.Series((sheet2.data[cn(sheet2.name, i-1, j+2)]+\
                        sheet2.data[cn(sheet2.name, i, j)])/2,name=cn('roe_ttm_denominator', i, j)))
                else:
                    series_list.append(pd.Series((sheet2.data[cn(sheet2.name, i-1+((j+3)//12), (j+3) % 12)] +\
                        sheet2.data[cn(sheet2.name, i, j)])/2,name=cn('roe_ttm_denominator', i, j)))
        self.feature_df=pd.concat([self.feature_df,*series_list],axis=1)
    def sector_filtering(self):
        first_level_ind = ['钢铁', '纺织服装', '轻工制造', '建材',
                    '传媒', '国防军工', '商贸零售', '银行', '交通运输', '农林牧渔']
        sec_level_ind = ['其他化学制品Ⅱ', '生物医药Ⅱ', '石油化工', '通用设备',
                    '建筑施工', '计算机软件', '电源设备', '煤炭开采洗选', '证券Ⅱ', '白色家电Ⅱ', '其他医药医疗', '乘用车Ⅱ',
                    '光学光电', '旅游及休闲', '电气设备', '汽车零部件Ⅱ',
                    '新能源动力系统', '发电及电网', '专用机械',
                    '化学制药', '中药生产', '保险Ⅱ', '稀有金属',
                    '半导体', '通信设备制造', '食品', '消费电子', '酒类', '房地产开发和运营']
        slide=self.feature_df['一级行业']=='钢铁'
        self.feature_df['行业']=self.feature_df['一级行业'].copy()
        for f_ind in first_level_ind:
            slide=slide | (self.feature_df['一级行业']==f_ind)
        for s_ind in sec_level_ind:
            slide=slide | (self.feature_df['二级行业']==s_ind)
            self.feature_df.loc[self.feature_df['二级行业']==s_ind,'行业']=s_ind
        self.feature_df=self.feature_df[slide]
    def listing_date_filtering(self,t):
        self.feature_df['signal']=self.feature_df['上市日期'].apply(lambda x: 0 if (x-t).days<365 else 1)
    def yoy_ratio(self, co, y, m):
        se=(self.feature_df[cn(co,y,m)]-self.feature_df[cn(co,y-1,m)])/self.feature_df[cn(co,y-1,m)]
        return se
    def qoq_ratio(self, co, y, m):
        if m == 6:
            se = (self.feature_df[cn(co, y, m)]-self.feature_df[cn(co, y, m-4)])/self.feature_df[cn(co, y, m-4)]
        elif m == 9:
            se = (self.feature_df[cn(co, y, m)]-self.feature_df[cn(co, y, m-2)])/self.feature_df[cn(co, y, m-2)]
        else:
            se = (self.feature_df[cn(co, y, m)]-self.feature_df[cn(co, y-1+(m+9)//12, (m+9)%12)]) / \
                self.feature_df[cn(co, y-1+(m+9)//12,(m+9)%12)]
        return se
    def mom_ratio(self, co, y, m):
        if m == 0:
            se = (self.feature_df[cn(co, y, m)]-self.feature_df[cn(co, y-1, 11)])/self.feature_df[cn(co, y-1, 11)]
        else:
            se = (self.feature_df[cn(co, y, m)]-self.feature_df[cn(co, y, m-1)])/self.feature_df[cn(co, y,m-1)]
        return se
class score():
    def __init__(self,df):
        self.score_df=pd.DataFrame(df.groupby(by='行业').count()['Code'])
        self.score_df.rename(columns={'Code':'Count'},inplace=True)
    def global_method(self,df,feature_1,feature_2,feature_name):
        self.score_df[feature_name]=df[feature_1]/df[feature_2]
    def signal_method(self,df,feature,signal_feature):
        self.score_df[signal_feature]=df[feature]/self.score_df['Code']
    #def cal_score(self):
    