"""
HR Attrition Prediction

- Logistic Regression (L2 regularization)
- Random Forest comparison
- Stratified train-test split
- Evaluation with ROC-AUC, F1-score

Author: sungjihye
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("/content/HR.csv")

"""# **1.EDA 및 Data manipulation**"""

df.head()

df.info()

df.describe().T

df['sales'].value_counts()

df['salary'].value_counts()

df.isnull().sum()

#satisfaction_level,last_evaluation,number_project,average_monthly_hours,Work_accident
import seaborn as sns
import math

cols=['satisfaction_level','last_evaluation','number_project','average_monthly_hours','Work_accident']

n_cols = 3
n_rows = math.ceil(len(cols)/n_cols)

fig,axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes= axes.flatten() #2d->1d

for idx, col in enumerate(cols):
  ax=axes[idx]

  #결측치를 제거한 분포 확인
  series= df[col].dropna()

  #변수가 이산형인지 연속형인지 판별  #value의 고유값이 10이하면 이산형/범주형
  if series.nunique() <= 10:
    #이산형/범주형
    sns.countplot(x=series,ax=ax)
    ax.set_title(f'{col}(Discrete / Categorical)')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

  else:
    #연속형
    sns.histplot(series, kde=True, bins=30, ax=ax)
    ax.axvline(series.mean(), color='r', linestyle='--', label='Maen')
    ax.axvline(series.median(), color='g', linestyle='--', label='Median')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.legend()
 #남는 subplot제거(몇개의 subplot이 필요한가를 기준으로 판단하는 코드)
for i in range(len(cols), n_rows * n_cols):
  #len(cols): 실제 사용한 subplot의 개수, n_rows * n_cols: 전체 subplot개수
  fig.delaxes(axes[i]) #i번째 subplot(axes객체)

plt.tight_layout()
plt.show()

"""#분포해석
##1.satisfaction_level(연속형)
분포가 완전히 대칭적이지 않고, 낮은 만족도 구간에 작은 봉우리 존재하므로, 평균으로 대체 시 "일반적인 직원"보다 낮은 값으로 왜곡 가능성이 있으므로 중앙값으로 대체
##2.last_evaluation(연속형)
분포가 비교적 고르고 평균과 중앙값이 거의 겹쳐져있으므로, 평균과 중앙값 어느쪽을 사용해도 무방하나 일관성을 위해 중앙값으로 대체
##3.number_project(이산형)
3,4 값에 몰려있고, 평균과 중앙값은 존재하지 않는 값으로 의미가 있는 최빈값으로 대체
##4.average_monthly_hours(연속형)
분포가 뚜렷하게 비대칭으로 장시간 근무자쪽 꼬리가 긴 형태로 분포가 한쪽으로 치우쳐진 형태이므로 평균값은 이에 영향을 받은 값이므로 중앙값으로 대체
##5.work_accident(범주형)
0이 1보다 압도적으로 많으므로 최빈값인 0으로 대체
"""

#average_monthly_hours 이상치 box plot으로 확인하기

n_col = ['satisfaction_level','last_evaluation','average_monthly_hours','time_spend_company']

for col in n_col:
  # IQR 계산
  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)
  IQR = Q3 - Q1
  lower = Q1 - 1.5 * IQR
  upper = Q3 + 1.5 * IQR
  outliers_count = df[(df[col] < lower) | (df[col] > upper)]

  print("IQR lower bound:", lower)
  print("IQR upper bound:", upper)
  print(f"{col} Number of IQR outliers:", len(outliers_count))


  # # boxplot
  # sns.boxplot(x=df[col], showfliers=True)

  # # outlier만 직접 scatter로 직접 표시(자동 flier(이상치점) 제거할 경우: showfliers=False)
  # # sns.scatterplot(
  # #     x=outliers[col],
  # #     y=[0] * len(outliers),
  # #     color='red',
  # #     s=40,
  # #     label='IQR Outliers'
  # # )

  # plt.title(f'{col} Boxplot with IQR Outliers')
  # plt.legend() #이상치가 있다면 그에대한 설명표시하는 코드
  # plt.show()

#subplot그리기
plt.figure(figsize=(8, 4))
bn_cols=2
bn_rows=math.ceil(len(n_col)/bn_cols)

fig,axes = plt.subplots(bn_rows, bn_cols, figsize=(12,4*bn_rows))
axes =axes.flatten()

for id, col in enumerate(n_col):
  ax=axes[id]
  sns.boxplot(x=df[col],ax=ax,showfliers=True)
  ax.set_title(col)

for i in range(len(n_col),bn_rows*bn_cols):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

"""average_monthly_hours 이상치 없음

time_spend_company의 경우 이상치라고 할수없음. 이는 장기 근속자를 의미하기 때문에 제거하면 안되는 의미 있는 값
"""

#satisfaction_level,last_evaaluation, average_monthly_hours -> 결측치 중앙값으로 대체
#number_project, Work_accident -> 최빈값으로 대체

nume_col=['satisfaction_level','last_evaluation','average_monthly_hours',]
cate_col=['number_project','Work_accident']

df[nume_col]=df[nume_col].apply(lambda x: x.fillna(x.median()))
df[cate_col]=df[cate_col].apply(lambda x: x.fillna(x.mode()[0]))

#결측치 대체 확인
df.isnull().sum()

#중복데이터 확인
duplicates= df.duplicated(keep=False)
print("중복 행 개수:", duplicates.sum())

#중복데이터 확인 및 제거
df=df.drop_duplicates()
print("중복 행 개수:", df.duplicated().sum())

"""#변수 정보
##연속형 변수
satisfaction_level: 만족도

last_evaluation: 성과 평가 점수

number_project: 맡은 프로젝트수

average_monthly_hours: 월평균 근무시간

time_spend_company: 근속 연수


---



##이산형/범주형변수
Work_accident: 산업재해

promotion_last_5years: 5년동안 승진여부

sales: 부서

salary:연봉 정도

left: 퇴직여부

##부서와 퇴직여부간 카이제곱 검정
"""

from scipy.stats import chi2_contingency

#crosstab 생성(빈도표생성)
contingency_table = pd.crosstab(
    df['sales'],
    df['left']
)

contingency_table

#카이제곱 독립성 검정
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

a = 0.05

if p_value < a:
  print("대립가설 채택: 부서와 퇴직여부는 독립이 아니다(연관이 있다)")
else:
  print("귀무가설 채택: 부서와 퇴직여부는 독립이다")

#검정 타당성 확인
expected_df=pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
    )
expected_df

"""부서와 퇴직여부 간의 독립성을 검정하기 위해 카이제곱 독립성 검정을 수행

결과) p-value가 0.05 미만 -> 두 변수 간에 통계적으로 유의한 연관성이 있음을 확인

*모든 셀의 기대빈도가 5 이상으로 확인되어 카이제곱 검정의 가정이 충족되었음을 확인

 이에 따라 이후 분석에서는 **부서별로 퇴직여부를 그룹화하여 분석을 진행**
"""

# 부서별 퇴직률 그룹화 #이진분류에서는 평균이 비율(퇴직X: 0, 퇴직O: 1)
sales_left_rate=df.groupby('sales')['left'].mean().sort_values(ascending=False).reset_index()
sales_left_rate

figure = sales_left_rate.plot(kind='bar', x='sales', y='left', figsize=(10, 6))
figure.set_xlabel('sales')
figure.set_ylabel('left')
figure.set_title('sales_left_rate')

plt.tight_layout()
plt.show()

"""부서별 퇴직률이 순위확인

부서별 퇴직률에 영향을 미치는 변수를 찾기 위해 Facet Boxplot(조건부 분포 비교)를 사용

퇴직률: hr> accounting = technical = support = sales > marketing = IT >> R&D = management


"""

sale_factor=['average_monthly_hours','satisfaction_level','number_project','last_evaluation','time_spend_company']

for f in sale_factor:
  g = sns.catplot(
    data=df,
    x='left',
    y= f,
    col='sales',
    kind='box',
    height=5,
    aspect=0.8,
    col_wrap=4
    )
  g.set_axis_labels("Attrition (0 = Stay, 1 = Leave)", f)
  g.set_titles("Department: {col_name}")
  plt.tight_layout()
  plt.show()

"""
Facet Boxplot 분석 결과, 대부분의 부서에서 퇴직자들은 재직자에 비해 평균 월 근무시간이 높고 만족도는 현저히 낮은 반면, 성과평가 점수와 근속연수는 상대적으로 높은 경향을 보였다.
이는 고성과·고투입 인력이 조직에 장기간 머문 이후 이탈하는 구조를 시사한다.

반면, 인사 부서(HR)의 경우 퇴직자들이 재직자보다 오히려 성과평가 점수와 근무시간이 낮게 나타나, 다른 부서들과 상이한 퇴직 패턴을 보였다.
이러한 결과를 통해 퇴직의 원인은 부서별로 상이할 가능성이 높다고 판단하였으며,
이에 따라 부서별 퇴직에 가장 큰 영향을 미치는 요인에 대한 근거를 추가적으로 탐색하고자 한다.

또한 현재 분석에 보상 수준이 포함되지 않았으므로, 급여 변수(salary)를 순서형 변수로 변환하여 추가 분석을 진행할 예정이다.

"""

#salary 순서형 변수로 변환
salary_map = {
    'low': 0,
    'medium': 1,
    'high': 2
}

df['salary_ord'] = df['salary'].map(salary_map)

#부서별 퇴직 여부에 따른 차이
summary = (
    df
    .groupby(['sales', 'left'])
    .agg(
        avg_hours=('average_monthly_hours', 'mean'),
        avg_satisfaction=('satisfaction_level', 'mean'),
        avg_projects=('number_project', 'mean'),
        avg_eval=('last_evaluation', 'mean'),
        avg_tenure=('time_spend_company', 'mean'),
        avg_salary=('salary_ord', 'mean'),
        count=('left', 'count')
    )
    .reset_index()
)

summary.T

"""
각 부서별로, 퇴직자와 재직자간 평균 차이가 가장 큰 변수 확인

이때, 변수마다 단위가 달라, 변수끼리 비교 불가하므로 단위에 독립적인 비교 지표로 Cohen's d 사용

d=(μ1−μ0)/s_pooled
=(퇴직자-재직자 평균 차이)/두 집단의 공동 표준 편차


 **목적) 효과크기를 기반으로 부서별 퇴직 요인에 대한 가설을 생성**"""

# Cohen's d 함수정의
def cohens_d(x0, x1):
    n0, n1 = len(x0), len(x1)
    s0, s1 = np.var(x0, ddof=1), np.var(x1, ddof=1)
    s_pooled = np.sqrt(((n0 - 1)*s0 + (n1 - 1)*s1) / (n0 + n1 - 2))
    return (np.mean(x1) - np.mean(x0)) / s_pooled


#부서별,변수별 effect size 계산
vars_to_check = [
    'average_monthly_hours',
    'satisfaction_level',
    'number_project',
    'last_evaluation',
    'time_spend_company',
    'salary_ord'
]

rows = []

for dept in df['sales'].unique():
    df_dept = df[df['sales'] == dept]

    stay = df_dept[df_dept['left'] == 0]
    leave = df_dept[df_dept['left'] == 1]

    for var in vars_to_check:
        d = cohens_d(stay[var], leave[var])
        rows.append({
            'sales': dept,
            'variable': var,
            'cohens_d': d,
            'abs_d': abs(d)
        })

effect_df = pd.DataFrame(rows)

#부서별 가장 영향력 큰 변수 찾기
top_effect = (
    effect_df
    .sort_values('abs_d', ascending=False)
    .groupby('sales')
    .first()
    .reset_index()
)

#확인
top_effect

"""부서별 퇴직 요인을 비교한 결과, 만족도 변수는 모든 부서에서 퇴직자와 재직자를 가장 강하게 구분하는 공통요인으로 나타났다. (|d| > 0.8)

이에 따라 부서간 차별적인 퇴직 매커니즘을 탐색하기 위해 만족도를 통제 변수로 간주하고, 그 외 변수들을 중심으로 추가 분석을 수행
"""

#부서별,변수별 2차요인 확인을 위해 effect sizef를 satisfaction_level제외하고 재계산
sec_check = [
    'average_monthly_hours',
    'number_project',
    'last_evaluation',
    'time_spend_company',
    'salary_ord'
]

rows = []

for dept in df['sales'].unique():
    df_dept = df[df['sales'] == dept]

    stay = df_dept[df_dept['left'] == 0]
    leave = df_dept[df_dept['left'] == 1]

    for sec in sec_check:
        d = cohens_d(stay[sec], leave[sec])
        rows.append({
            'sales': dept,
            'variable': sec,
            'cohens_d': d,
            'abs_d': abs(d)
        })

effect_df = pd.DataFrame(rows)

#부서별 가장 영향력 큰 변수 찾기
second_effect = (
    effect_df
    .sort_values('abs_d', ascending=False)
    .groupby('sales')
    .first()
    .reset_index()
)

#확인
second_effect

"""2차 요인이 약한지 중간인지 강한지 해석하기 위해 범주화

0.2 ~ 0.5  → Small

0.5 ~ 0.8  → Medium

≥ 0.8      → Large
"""

#d 값에 따라 범주화
bins = [0, 0.2, 0.5, 0.8, np.inf]
labels = ['Negligible', 'Small', 'Medium', 'Large']

second_effect['effect_size_level'] = pd.cut(
    second_effect['abs_d'],
    bins=bins,
    labels=labels,
    right=False   # 경계값 포함 방식 명확화
)

second_effect

"""##결과 해석

만족도를 공통 1차 요인으로 간주하고 효과크기 기준의 탐색적 분석을 수행한 결과,
부서별로 퇴직자와 재직자를 구분하는 2차 요인이 상이할 가능성이 관찰되었다.

특히 management 부서에서는 급여 수준(salary)이 상대적으로 큰 효과크기를 보였으며,
이는 급여 수준에 따라 퇴직 여부의 분포가 달라질 가능성을 가설로 설정하게 하였다.
이에 따라 해당 부서를 대상으로 급여 수준과 퇴직 여부 간의 관계를
통계적으로 검증할 필요가 있다고 판단하였다.

또한 R&D, HR, product management, support, technical 부서에서는
근무 연수(time_spend_company)가 중간 수준의 효과크기를 보였으며,
근속 단계에 따른 퇴직 분포 차이가 존재할 가능성을 가설로 설정하였다.
이러한 가설을 검증하기 위해 부서별로 근무 연수에 따른 퇴직 여부의 차이를
ANOVA 또는 χ² 검정을 통해 확인하고자 한다.

반면 IT, accounting, sales 부서에서는
2차 요인의 효과크기가 상대적으로 작게 나타나,
만족도가 퇴직 여부를 설명하는 주요 요인일 가능성을 가설로 설정하였다.
이 역시 통계적 검정을 통해 추가 요인의 유의성을 확인할 예정이다.


---

# **2.가설검정 및 통계적 검증**

##Management 부서 — 급여(salary)
분석 배경

2차 요인으로 salary가 Large effect

범주형 변수 (low / medium / high)

[χ² test (독립성 검정)]

귀무가설 (H₀)

Management 부서에서 급여 수준과 퇴직 여부는 서로 독립이다.

대립가설 (H₁)

Management 부서에서 급여 수준과 퇴직 여부는 서로 독립이 아니다.

➡ 급여 수준에 따라 퇴직 비율이 달라지는지 검증


---

##R&D / HR / Product Management / Support / Technical — 근무 연수(time_spend_company)
분석 배경

2차 요인으로 time_spend_company가 Medium effect

연속형 변수

[ANOVA]

귀무가설 (H₀)

해당 부서에서 퇴직 여부에 따른 평균 근무 연수(time_spend_company)는 차이가 없다.

대립가설 (H₁)

해당 부서에서 퇴직 여부에 따른 평균 근무 연수(time_spend_company)는 차이가 있다.

➡ 근속 단계에 따른 퇴직 패턴 차이 검증


---

##IT / Accounting / Sales — 만족도 중심 구조
분석 배경

2차 요인 효과 작음 (Small)

만족도가 핵심 요인일 가능성

[ANOVA]

귀무가설 (H₀)

해당 부서에서 퇴직 여부에 따른 평균 만족도(satisfaction_level)는 차이가 없다.

대립가설 (H₁)

해당 부서에서 퇴직 여부에 따른 평균 만족도(satisfaction_level)는 차이가 있다.

➡ 만족도 단독 설명력 검증


---
"""

#1. Management 부서와 salary 간의 chi2 test
#H0: Managenent부서에서 급여수준과 퇴직 여부는 서로 독립이다.

df_man=df[df['sales']=='management']

contingency=pd.crosstab(
    df_man['salary'],
    df_man['left']
)

contingency

chi2,p_value,dof,expected = chi2_contingency(contingency)

print(f"Chi2 statistic: {chi2:.4f}")
print(f"p-value: {p_value}")
print(f"Degrees of freedom: {dof}")

a = 0.05

if p_value < a:
  print("대립가설 채택:Management부서에서 급여수준과 퇴직여부는 독립이 아니다.")
else:
  print("귀무가설 채택: Management부서에서 급여수준과 퇴직여부는 독립이다")

expeted_man=pd.DataFrame(
    expected,
    index=contingency.index,
    columns=contingency.columns
)

expeted_man

"""카이제곱 독립성 검정 결과(p < 0.05), Management 부서에서 급여 수준과 퇴직 여부는 통계적으로 유의한 연관성이 있는 것으로 나타났다. 또한 기대빈도 조건을 충족하여 검정의 타당성을 확인하였다. 이는 Management 부서에서 급여 수준이 퇴직 여부를 설명하는 주요 요인일 가능성을 시사한다."""

df['sales'].unique()

#RandD / hr / product_mng / support / technical — 근무 연수(time_spend_company)
#ANOVA
#H₀ :해당 부서에서 퇴직 여부에 따른 평균 근무 연수(time_spend_company)는 차이가 없다.
from scipy.stats import f_oneway

target = ['RandD','hr','product_mng','support','technical']

anova_results = []

for dept in target:
  df_target = df[df['sales'] == dept]

  stay = df_target[df_target['left'] == 0]['time_spend_company']
  leave = df_target[df_target['left'] == 1]['time_spend_company']

  f_stat,p_value = f_oneway(stay,leave)

  anova_results.append({
      'department': dept,
      'F_statistic': f_stat,
      'p_value': p_value,
      'mean_stay':stay.mean(),
      'mean_leave':leave.mean(),
      'n_stay':len(stay),
      'n_leave':len(leave)
      })

  a=0.05

  if p_value < a:
    print(f"{dept} 부서에서 퇴직 여부에 따른 평균 근무 연수에 차이가 있다.")
  else:
    print(f"{dept} 부서에서 퇴직 여부에 따른 평균 근무 연수에 차이가 없다.")

anova_df = pd.DataFrame(anova_results)

anova_df

"""ANOVA 결과(p < 0.05), 해당 부서에서 퇴직 여부에 따라 평균 근무 연수에 통계적으로 유의한 차이가 있는 것으로 나타났다. 이는 근속 단계에 따라 퇴직 패턴이 달라질 가능성을 시사한다."""

#IT / Accounting / Sales — 만족도 중심 구조
#ANOVA
#H₀:해당 부서에서 퇴직 여부에 따른 평균 만족도(satisfaction_level)는 차이가 없다.

target_depts = ['IT', 'accounting', 'sales']

anova_results = []

for dept in target_depts:
    df_dept = df[df['sales'] == dept]

    stay = df_dept[df_dept['left'] == 0]['satisfaction_level']
    leave = df_dept[df_dept['left'] == 1]['satisfaction_level']

    f_stat, p_value = f_oneway(stay, leave)

    anova_results.append({
        'department': dept,
        'F_statistic': f_stat,
        'p_value': p_value,
        'mean_stay': stay.mean(),
        'mean_leave': leave.mean(),
        'n_stay': len(stay),
        'n_leave': len(leave)
    })

    a=0.05

    if p_value < a:
      print(f"{dept} 부서에서 퇴직 여부에 따라 평균 만족도에 차이가 있다.")
    else:
      print(f"{dept} 부서에서 퇴직 여부에 따라 평균 만족도에 차이가 없다.")

anova_df = pd.DataFrame(anova_results)

anova_df

"""ANOVA 결과(p < 0.05), 해당 부서에서 퇴직 여부에 따라 평균 만족도에 통계적으로 유의한 차이가 있는 것으로 나타났다.
이는 만족도가 퇴직 여부를 설명하는 핵심 요인일 가능성을 시사한다.

---

# **3.퇴직여부 예측 모델링**
통계적 검정 결과를 종합하면 다음과 같은 구조가 관찰되었다.

**만족도(satisfaction_level)** 는 모든 부서에서 공통적으로 유의한 요인으로 확인되었다.

**급여 수준(salary)** 과 **근무 연수(time_spend_company)** 는 부서에 따라 그 영향력이 상이하게 나타났다.

이는 퇴직 메커니즘이 부서별로 동일하게 작동하지 않으며,
공통 요인과 부서별 차별 요인이 함께 작용하는 구조임을 시사한다.

위의 검정 결과는 이후 퇴직 여부 예측 모델의 설계 방향을 결정하는 근거로 활용하였다.

만족도는 모든 부서에서 공통 설명 변수로 포함

급여 수준과 근무 연수는 부서별로 상이한 영향을 반영하기 위해
부서 변수와의 상호작용(interaction) 후보 변수로 고려 (interaction을 고려하면, 로지스틱 회귀에서 각 부서별로 기울기와 절편이 달라짐)

이를 바탕으로, 다음 단계에서는
부서별 퇴직 메커니즘의 차이를 반영한 이진 분류 모델을 구축하고
예측 성능을 중심으로 모델을 평가하고자 한다.

이때 모델링 단계에서는 다변량 환경에서 변수 간 중복 설명을 고려하기 위해 L2 정규화( Ridge)가 적용된 로지스틱 회귀 모델을 사용할 것이며,
비선형 관계를 자동으로 학습하는 Random Forest 모델과의 성능 비교를 통해 예측력을 추가로 검증할 것이다.


\\
"""

df['left'].value_counts()
#left 클래스 불균형 올 가능성을 고려하여 데이터 분리시 층화 샘플링 적용(stratified sampling)

from sklearn.model_selection import train_test_split

x = df.drop(columns= ['left'])
y = df['left']

#dataset split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,
                                                    stratify=y, random_state=42)

#Logistix Regression은 StandardScaler 필수
#sales 는 범주형 변수이므로 one-hot encoding 적용

#ColumnTransformer로 변수별 전처리 적용
#코드 스니펫
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_features),      # 연속형들만 표준화
#         ('cat', OneHotEncoder(drop='first'), categorical_features)  # 범주형만 원핫
#     ]
# )

#pipeline 코드 스니펫
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LogisticRegression

# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),   # ColumnTransformer
#     ('interaction', PolynomialFeatures(
#         degree=2,
#         interaction_only=True,
#         include_bias=False
#     )),
#     ('classifier', LogisticRegression(
#         penalty='l2',
#         solver='lbfgs',
#         max_iter=1000
#     ))]
# )

#PolynomialFeatures(
#     degree=2,
#     interaction_only=True,
#     include_bias=False
# )
#“기울기(slope)만 달라지게 하는 interaction만 추가한다”=“부서별 효과 조절”

#ColumnTransformer 정의
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = [
    'satisfaction_level',
    'time_spend_company',
    'average_monthly_hours',
    'number_project',
    'last_evaluation',
    'salary_ord'
]

categorical_features = ['sales']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

#Interaction + Logistic Regression Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('interaction', PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False
    )),
    ('classifier', LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=1000
    ))]
)

# 모델 학습
model.fit(x_train, y_train)

#모델 평가
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

"""Logistic Regression (L2 Regularization, Ridge)

본 분석에서는 다변량 환경에서 변수 간 중복 설명(multicollinearity)으로 인한 계수 불안정을 완화하기 위해 L2 정규화가 적용된 로지스틱 회귀 모델을 사용하였다.

해당 모델은 소수 클래스(퇴직자)에 대해 precision 0.85, recall 0.84, F1-score 0.85를 기록하였으며, ROC-AUC는 0.96으로 높은 분류 성능을 보였다. 

이는 선형 가정 하에서도 주요 예측 신호가 충분히 포착되고 있음을 의미한다.

특히 클래스 불균형 상황에서도 macro average 지표가 안정적으로 유지되어, 모델이 특정 클래스에 편향되지 않고 균형 잡힌 예측을 수행하고 있음을 확인하였다. 

본 모델은 해석 가능성과 예측 성능 간의 균형 측면에서 합리적인 기준 모델(baseline)로 활용 가능하다.
"""

x_train

#Random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42, class_weight='balanced') #블균형데이터 대응

rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", rf)
])

rf_model.fit(x_train, y_train)

y_pred_rf = rf_model.predict(x_test)
y_proba_rf = rf_model.predict_proba(x_test)[:,1]

print(classification_report(y_test, y_pred_rf))
print("ROU_AUC:", roc_auc_score(y_test, y_proba_rf))

"""Random Forest Classifier

로지스틱 회귀 모델의 선형 가정 한계를 보완하기 위해, 비선형 관계 및 변수 간 상호작용을 자동으로 학습하는 Random Forest 모델을 추가적으로 적용하였다.

Random Forest 모델은 소수 클래스(퇴직자)에 대해 precision 0.99, recall 0.92, F1-score 0.95를 기록하여, 퇴직자 탐지 성능이 전반적으로 향상되었음을 확인하였다. 정확도는 0.98로 상승하였으며, 

다수·소수 클래스 모두에서 높은 분류 성능을 보였다.

이는 일부 변수 조합에서 비선형적 패턴이 존재함을 시사하며, 보다 복잡한 의사결정 경계를 통해 오분류를 감소시킬 수 있음을 보여준다.

다만 모델 구조의 특성상 해석 가능성은 상대적으로 제한된다.

※ 참고: ROC-AUC가 로지스틱 회귀와 동일하게 나타난 것은, 두 모델 모두 클래스 순위 분리 능력은 유사하나, 결정 경계의 형태 차이로 실제 분류 성능(precision/recall)에 차이가 발생했음을 의미한다.

---

# **최종 결론**

본 분석에서는 L2 정규화가 적용된 로지스틱 회귀 모델과 Random Forest 모델을 비교함으로써, 퇴직 여부 예측에 있어 선형 및 비선형 모델의 성능 차이를 검증하였다.

로지스틱 회귀 모델은 높은 ROC-AUC(0.96)와 안정적인 분류 성능을 통해, 주요 예측 신호가 비교적 선형적인 구조를 갖고 있음을 보여주었으며, 해석 가능성 측면에서 강점을 가진다. 

반면, Random Forest 모델은 소수 클래스에 대한 재현율과 F1-score를 추가적으로 개선하여, 변수 간 비선형 상호작용이 일부 존재함을 시사한다.

이에 따라 설명 목적 및 정책 해석이 중요한 경우 로지스틱 회귀 모델을, 예측 정확도 극대화가 요구되는 경우 Random Forest 모델을 선택할 수 있는 근거를 확보하였다. 
"""

