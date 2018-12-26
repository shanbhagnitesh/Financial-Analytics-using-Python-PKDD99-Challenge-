# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 19:28:38 2018


"""


############################################################################################
####################         IMPORT PACKAGES            #################################### 
############################################################################################

import pandas as pd

from datetime import datetime

import numpy as np

from matplotlib import pyplot as plt

############################################################################################
####################         IMPORT DATASETS            #################################### 
############################################################################################

path = "C:\\Users\\mrudrappa\\Desktop\\python group assignment\\MBD2018_FP_GroupAssignment_FinancialData\\data_berka\\"
client = pd.read_csv(path + "client.asc",sep=";")
account = pd.read_csv(path + "account.asc",sep=";")
disp = pd.read_csv(path + "disp.asc",sep=";")
card = pd.read_csv(path + "card.asc",sep=";")
district = pd.read_csv(path + "district.asc",sep=";")
loan = pd.read_csv(path + "loan.asc",sep=";")
order = pd.read_csv(path + "order.asc",sep=";")
trans = pd.read_csv(path + "trans.asc",sep=";")


# to check duplicates client id in disp.
pd.concat(g for _, g in disp.groupby("account_id") if len(g) > 1)

############################################################################################
####################         DEMOGRAPHIC DATASETS            ############################### 
############################################################################################

# summary of district 
summary = district.describe()    
summary = summary.transpose() 
print(summary) 



# head 
district.head(6)

# Rename the columns
district = district.rename(columns={"A1":"district_id","A2":"district_name","A3":"region","A4":"no_of_inhabitants",
                         "A5":"munci_lt_499","A6":"munci_500_1999","A7":"munci_2000_9999",
                         "A8":"munci_gt_10000","A9":"no_of_cities","A10":"ratio_of_urban_inhab",
                         "A11":"district_averagr_salary","A12":"unemployement_rate_1995","A13":"unemployement_rate_1996",
                         "A14":"prob_enterpreneur","A15":"prob_commited_crimes_1995","A16":"prob_commited_crimes_1996"})

#converted objects to numeric 
district = district.convert_objects(convert_numeric =True)

#fill nan values by mean
district = district.fillna(district.mean())

#find the probabilities
district["prob_enterpreneur"] = district["prob_enterpreneur"] / 1000
district["prob_commited_crimes_1995"] = district["prob_commited_crimes_1995"] / district["no_of_inhabitants"]
district["prob_commited_crimes_1996"] = district["prob_commited_crimes_1996"] / district["no_of_inhabitants"]

# check the data types of variables
district.dtypes                      
                
############################################################################################
####################         CLIENT DATASETS                 ############################### 
####################  WE HAVE ASSUMED CURRENT YEAR AS 1999   ###############################
############################################################################################

# Transform the birth day into year
client['birth_year'] = client['birth_number'].transform(lambda bn: int('19' + str(bn)[:2]))

# Transform the birth day into day
client['birth_day'] = client['birth_number'].transform(lambda bn: int(str(bn)[4:6]))
# Age 
client['age'] = 1999 - client['birth_year']

# Age group
client['age_group'] = client['age'] // 10 * 10

# Function to extract birth month and gender
def to_month_gender(birth_number):
    
    s = str(birth_number)
    birth_month = int(s[2:4])
    
    if birth_month > 50:
        gender = "F"
        birth_month = birth_month - 50
    else:
        gender = 'M'
        
    return pd.Series({'birth_month':birth_month, 'gender':gender})

client[['birth_month', 'gender']] = client['birth_number'].apply(to_month_gender)

# derive variable birth date
client['birth_date'] = client.apply(lambda row: datetime(row['birth_year'], row['birth_month'], row['birth_day']), axis=1)
client['birth_date'] = pd.to_datetime(client['birth_date']).dt.date

# drop unwanted columns
client = client.drop(['birth_year', 'birth_number','birth_month','birth_day'], axis=1)

client.dtypes


############################################################################################
####################         CREDIT CARD DATASET              ############################## 
############################################################################################

card['issued'] = pd.to_datetime(card['issued']).dt.date
card = card.rename(columns = {"type":"credit_card_type","issued":"credit_card_issue_date"})
card['credit_card_issue__year'] = card['credit_card_issue_date'].transform(lambda bn: int(str(bn)[:4]))
############################################################################################
####################         LOAN DATASET                     ############################## 
############################################################################################

loan.info()

#change the date format
loan[['date']] = loan[['date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[4:],s[2:4],s[0:2]))

#drop loan id
loan = loan.drop(["loan_id"], axis=1)
loan['status'].unique()

# change the description of the field status 
loan['status'] = loan['status'].map({'A': 'Contract finished without Issue', 'B': 'Contract Finished Loan Not Paid'
                                      , 'C': 'Contract Running ok so far', 'D' : 'Contract Running Debt'})
# rename the columns
loan = loan.rename(index=str, columns={"date": "LoanIssuedDate", "amount": "LoanAmount"
                                       ,"duration": "LoanDuration","payments": "LoanPayment","status": "LoanStatus"})

    
    
############################################################################################
####################         ORDER DATASET                     ############################# 
############################################################################################

order.count()
order['account_to'].nunique()
order = order.drop(["order_id"], axis=1)
order = order.drop(["bank_to"], axis=1)
order = order.drop(["account_to"], axis=1)
order['k_symbol'].unique().tolist()


order.loc[order['k_symbol'] == " ", 'k_symbol'] = 'Other'


order = order.pivot_table(index=['account_id'],columns='k_symbol',values='amount',fill_value=0)
order.columns.name = None
order = order.reset_index()


order['account_id'].nunique()


# As the Above is also equal to 3758 = Number of records in Order so that means all Account_id is unique.


############################################################################################
####################        ACCOUNT DATASET                    ############################# 
############################################################################################

account['account_id'].nunique()

# account year tofind length of relation
account['account_created_year'] = account['date'].transform(lambda bn: int('19' + str(bn)[:2]))

# length of relation
account['length_of_relation'] = 1999 - account['account_created_year']

account[['date']] = account[['date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[2:4],s[4:],s[0:2]))

account['frequency'].unique().tolist()
account['frequency'] = account['frequency'].map({'POPLATEK MESICNE': 'Monthly Issuance',
                                                 'POPLATEK TYDNE'  : 'Weekly Issuance'
                                                 ,'POPLATEK PO OBRATU' : 'Issuance after Trans'})   
    
account = account.rename(index=str, columns={"date": "AccountCreatedDate"})


############################################################################################
####################       TRANSACTION DATASET                    ########################## 
############################################################################################
  
trans = trans.drop(["trans_id"], axis=1)
# Changing into Valid Dateformat
trans[['date']] = trans[['date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[2:4],s[4:],s[0:2]))
# Dropping un neccesary details
trans = trans.drop(["account"], axis=1)
trans = trans.drop(["bank"], axis=1)
#Changing Values to more understandle ones
trans['type'] = trans['type'].map({'VYBER': "Debit",'PRIJEM':"Credit", 'VYDAJ':"Debit"})

trans['operation'] = trans['operation'].map({'VYBER KARTOU' : "Credit Card Withdrawal",
                                             'VKLAD'        : "Credit In Cash", 
                                             'PREVOD Z UCTU': "Collection from another Bank",
                                             'VYBER'        : "Withdrawal In Cash",
                                             'PREVOD NA UCET' : "Remittance to another Bank"})

trans = trans.drop(["k_symbol"], axis=1)
#Replacing nan with Other
trans = trans.replace(np.nan, "Other")
trans['amount1'] = trans.amount
#Bringing the Type as Columns to calculate Average per month later
trans = trans.pivot_table(index=['account_id','date','operation','balance','amount1'],columns='type',values='amount',fill_value=0)
trans.columns.name = None
trans = trans.reset_index()

#Pivoting Operation also as columns

trans = trans.pivot_table(index=['account_id','date','balance','Credit','Debit'],columns='operation',values='amount1',fill_value=0)
trans.columns.name = None
trans = trans.reset_index()
############## Average Month End Balances 
# Logic is For each month end to find the account Balances so from that we can get an idea about what are his savings
#at monthe end

# for that month end row is only retained along with its balance and for that the Group by condition has been applied
# to get Mean and Median Balance at Month end.
transbalance = trans[['account_id', 'date', 'balance']]
transbalance['monthyear']  = transbalance[['date']].applymap(str).applymap(lambda s: "{}{}".format(s[6:],s[0:2])) 
transbalance['day']  = transbalance[['date']].applymap(str).applymap(lambda s: "{}".format(s[3:5])) 
transbalance = transbalance.drop(["date"], axis=1)

transbalance = transbalance.sort_values(['account_id', 'monthyear','day'], ascending=[True, True,False])

def f(s):
    s2 = pd.Series(0, index=s.index)
    s2.iloc[-1] = 1
    return s2

transbalance["lastMark"] = transbalance.groupby(['account_id','monthyear'])['day'].apply(f)

transbalance = transbalance[transbalance.lastMark == 1]

transbalance = transbalance.drop(["monthyear"], axis=1)
transbalance = transbalance.drop(["day"], axis=1)
transbalance = transbalance.drop(["lastMark"], axis=1)

transbalance = transbalance.groupby('account_id', as_index=False).agg({'balance': ['mean', 'median']})

transbalance.columns  = ['account_id','Balance_Mean_per_Month','Balance_Median_per_Month']

################### 
# Here the Monthly all the different amounts are summed and after we get 1 row per month for each account id
# then Month wise averages are calculated.
# So we get an idea about his Monthly income Monthly expenditure etc from that.

transamounts = trans
transamounts = transamounts.drop(["balance"], axis=1)


transamounts['monthyear']  = transamounts[['date']].applymap(str).applymap(lambda s: "{}{}".format(s[6:],s[0:2])) 
transamounts = transamounts.drop(["date"], axis=1)

transamounts = transamounts.sort_values(['account_id', 'monthyear'], ascending=[True, True])

transamounts = transamounts.groupby(['account_id','monthyear']).sum().reset_index()

transamounts = transamounts.drop(["monthyear"], axis=1)
transamounts = transamounts.groupby(['account_id']).mean().reset_index()


trans = pd.merge(transamounts,transbalance, on=['account_id'])

# Rename columns
trans = trans.rename(columns={"Credit":"Avg_Monthly_Credit","Debit":"Avg_Monthly_Debit",
                              "Other":"Other_Transactions","Balance_Mean_per_Month":"Avg_Monthly_Balance",
                              "Median_Balance_per_Month":"Median_Monthly_Balance"})


# MONTHLY SAVINGS
trans['Avg_Monthly_Savings'] =  trans["Avg_Monthly_Credit"] - trans["Avg_Monthly_Debit"]

 
##########################################################################################
#merging all datasets
##########################################################################################

client_disp = pd.merge(client, disp, on='client_id', how='left')

client_district = pd.merge(client_disp, district, on='district_id', how='left')

client_final = pd.merge(client_district, card, on='disp_id', how='left')

account_trans = pd.merge(account, trans, on='account_id', how='left')

account_loan = pd.merge(account_trans, loan, on='account_id', how='left')

account_final = pd.merge(account_loan, order, on='account_id', how='left')

Client_Base_Table = pd.merge(client_final, account_final, on='account_id', how='left')

# cleaning NAN values
Client_Base_Table = Client_Base_Table.replace(np.nan, 0)     
Client_Base_Table['LoanStatus'].replace([0],['Not Applicable'],inplace=True)
Client_Base_Table['credit_card_type'].replace([0],['Not Applicable'],inplace=True)

Client_Base_Table.dtypes

##############################################################################################
# Visualisation
###########################################################################################


# Age Group that asked loan for the most
t = Client_Base_Table[Client_Base_Table.LoanStatus != 'Not Applicable']
t1 = t.groupby('age_group')['LoanStatus'].agg('count')
t1.plot.bar()
plt.ylabel('Number of time Loan asked')
print(t1)
plt.show()

# Loan status in each gender
t2 = pd.pivot_table(t, values='client_id', index='gender', columns='LoanStatus', aggfunc='count', fill_value=0)
t2.plot.bar()
print(t2)
plt.show()

# loan status in each gender group
t3=pd.pivot_table(t, values='client_id', index='age_group', columns='LoanStatus', aggfunc='count', fill_value=0)
t3.plot.bar()
print(t3)
plt.show()

# loan status in each group of Length of relationship
t4 = pd.pivot_table(t, values='client_id', index='length_of_relation', columns='LoanStatus', aggfunc='count', fill_value=0)
t4.plot.bar()
print(t4)
plt.show()


# Average monthly balance vs credit card
c = Client_Base_Table[Client_Base_Table.credit_card_type != 'Not Applicable']
c1 = c.groupby('credit_card_type')['Avg_Monthly_Balance'].agg('mean')
c1.plot.bar()
plt.ylabel('Avg Monthly Balance')
print(c1)
plt.show()


# Average monthly savings of clients in each region

c2 = Client_Base_Table.groupby('region')['Avg_Monthly_Savings'].agg('mean')
c2.plot.bar()
plt.ylabel('Avg Monthly Savings')
print(c2)
plt.show()

# Number of clients Yearly wise
c3 = Client_Base_Table.groupby('account_created_year')['client_id'].agg('count')
c3.plot.line()
print(c3)
plt.ylabel('Number Of Clients')
plt.show()

# Number of credit card issued Yearly wise
c5 = Client_Base_Table[Client_Base_Table.credit_card_issue__year != 0]
c4 =c5.groupby('credit_card_issue__year')['client_id'].agg('count')
c4.plot.line()
plt.ylabel('Number Of credit cards issued')
plt.show()

# Data to plot Types of Transactions
another_bank = round(sum(Client_Base_Table['Collection from another Bank']),2)
CreditCardWithdrawal = round(sum(Client_Base_Table['Credit Card Withdrawal']),2)
Credit_In_Cash = round(sum(Client_Base_Table['Credit In Cash']),2)
Other_Transactions = round(sum(Client_Base_Table['Other_Transactions']),2)
RemittanceotherBank = round(sum(Client_Base_Table['Remittance to another Bank']),2)
WithdrawalInCash = round(sum(Client_Base_Table['Withdrawal In Cash']),2)

labels = 'Coll_from_Other_Banks', 'CC Withdrawals', 'Credit_In_Cash', 'Other_Transactions', 'RemittanceOtherBank','WithdrawalInCash'
sizes = [another_bank, CreditCardWithdrawal, Credit_In_Cash, Other_Transactions,RemittanceotherBank,WithdrawalInCash]
colors = ['yellowgreen', 'red', 'lightskyblue', 'black','gold', 'pink']
explode = (0, 0.1, 0, 0.1, 0, 0.1)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

####################################################################

import seaborn
#Avg Monthly Savings
seaborn.distplot(Client_Base_Table['Avg_Monthly_Savings'], bins=20)
plt.show()

#Avg Monthly Credit
seaborn.distplot(Client_Base_Table['Avg_Monthly_Credit'], bins=20)
plt.show()

#Avg Monthly Debit
seaborn.distplot(Client_Base_Table['Avg_Monthly_Debit'], bins=20)
plt.show()


#####################################################################
# Summary Statistics


new = Client_Base_Table[['Avg_Monthly_Credit', 'Avg_Monthly_Debit', 'Avg_Monthly_Balance','Balance_Median_per_Month','Avg_Monthly_Savings','LoanAmount']].copy()


new = new.describe()
print(new)


####################################################################
# Total Amount of Transactions based on K-Type

LEASING = round(sum(Client_Base_Table['LEASING']),2)
Other = round(sum(Client_Base_Table['Other']),2)
POJISTNE = round(sum(Client_Base_Table['POJISTNE']),2)
SIPO = round(sum(Client_Base_Table['SIPO']),2)
UVER = round(sum(Client_Base_Table['UVER']),2)

labels = 'LEASING', 'Other', 'POJISTNE', 'SIPO', 'UVER'
sizes = [LEASING, Other, POJISTNE, SIPO,UVER]
colors = ['yellowgreen', 'red', 'lightskyblue', 'pink','gold']
explode = (0, 0.1, 0, 0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

####################################################################


Client_Base_Table.to_csv('Client_Base_Table.csv', sep=',')
Client_Base_Table.to_csv('Client_Base_Table', sep='\t')
