from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from werkzeug.utils import secure_filename
import os
import io

# 앞에서 작성한 유전 알고리즘 코드를 이 파일에 추가하거나 별도의 모듈로 가져옵니다.

# 조 구성 함수
from collections import defaultdict
import numpy as np

def create_groups(df):
    # 조를 저장할 딕셔너리 생성
    groups = defaultdict(list)

    # 조건에 따라 데이터프레임 정렬
    sorted_df = df
    #sorted_df = df.sort_values(by=['조장 희망 여부', '음주 호불호', '성별'], ascending=[False, True, True])

    # 조 구성
    group_number = 1
    for index, row in sorted_df.iterrows():
        added_to_group = False
        temp_group_number = 1

        while not added_to_group:
            # 현재 조에 후보자를 추가할 수 있는지 검사하고 조건을 만족하면 추가
            if check_conditions(groups[temp_group_number], row):
                groups[temp_group_number].append(row)
                added_to_group = True
            else:
                # 다음 조로 이동하거나, 모든 조를 검사한 경우 새로운 조를 생성
                temp_group_number += 1
                if temp_group_number > group_number:
                    group_number = temp_group_number

    return groups

# 음주 호불호 검사
def check_drinking_preference(current_group, candidate):
    for member in current_group:
        if member['음주 호불호'] == '호' and candidate['음주 호불호'] == '불호':
            return False
        if member['음주 호불호'] == '불호' and candidate['음주 호불호'] == '호':
            return False
    return True

# 나이 표준편차 검사
def check_age_std(current_group, candidate):
    ages = [member['나이'] for member in current_group]
    ages.append(candidate['나이'])
    if candidate['조원 나이대 희망'] == '비슷한 나이대 선호' and np.std(ages) > 1.5:
        return False
    return True

# 성비 검사
def check_gender_ratio(current_group, candidate):
    male_count = sum([1 for member in current_group if member['성별'] == '남'])
    female_count = sum([1 for member in current_group if member['성별'] == '여'])
    total_count = male_count + female_count

    if total_count >= 7:
        return False

    if candidate['성별'] == '남':
        male_count += 1
    else:
        female_count += 1

    if not (female_count <=4 and male_count <=4):
        return False

    return True

# 최소/최대 인원 검사
def check_min_max_members(current_group, candidate):
    if len(current_group) >= 7:
        return False
    return True

# 조장 여부 검사
def check_leader_preference(current_group, candidate):
    leaders = [member for member in current_group if member['조장 희망 여부'] == '희망']
    if (candidate['조장 희망 여부'] == '희망') and len(leaders) > 0: 
        return False
    return True

def check_conditions(current_group, candidate, drink = True, age = True, gender = True, num = True, leader = True):
    if drink == True:
        if not check_drinking_preference(current_group, candidate):
            return False
    if age == True:
        if not check_age_std(current_group, candidate):
            return False
    if gender == True:
        if not check_gender_ratio(current_group, candidate):
            return False
    if num == True:
        if not check_min_max_members(current_group, candidate):
            return False
    if leader == True:
        if not check_leader_preference(current_group, candidate):
            return False
    return True

# 결과 출력 함수
def print_result(groups):
    for group_number, members in groups.items():
        print(f"Group {group_number}:")
        for member in members:
            print(f"""이름: {member['이름']}, 나이: {member['나이']}, 음주 호불호: {member['음주 호불호']}, 나이대 선호: {member['조원 나이대 희망']}, 성별: {member['성별']}, 조장: {member['조장 희망 여부']}""")
        print()

# 결과를 DataFrame에 할당
def assign_groups_to_df(df, groups):
    df['Group'] = -1
    for group_number, members in groups.items():
        for member in members:
            idx = df[df['이름'] == member['이름']].index[0]
            df.at[idx, 'Group'] = group_number
    return df

import csv
import pandas as pd

# CSV 파일 읽기
def read_csv(file_name):
    df = pd.read_csv(file_name)
    return df

# 데이터 전처리
def preprocess_data(df):
    # 나이 str -> int 변환
    df['나이'] = df['나이'].astype(int)
    return df

import random


# 초기 인구 생성
def create_initial_population(df, population_size):
    population = []
    for _ in range(population_size):
        shuffled_df = df.sample(frac=1, random_state=random.randint(1, 1000)).reset_index(drop=True)
        groups = create_groups(shuffled_df)
        population.append(groups)
    return population

# 적합도 함수
def fitness_function(groups):
    total_score = 0
    weight_age_std = 1
    weight_gender_balance = 3
    weight_drinking = 5
    weight_leader = 1
    
    all_members = set()
    duplicated_members = set()

    for _, group in groups.items():
        group_size = len(group)
        
        # 나이 표준편차
        age_std = np.std([member['나이'] for member in group])

        # 성별 비율
        num_males = sum([member['성별'] == '남' for member in group])
        num_females = group_size - num_males
        gender_balance = abs(num_males - num_females)

        # 음주 호불호
        drinkers = sum([member['음주 호불호'] == '호' for member in group])
        non_drinkers = sum([member['음주 호불호'] == '불호' for member in group])
        drinking_issue = 0 if drinkers == 0 or non_drinkers == 0 else 1

        # 조장 희망자
        leaders = sum([member['조장 희망 여부'] in ['희망', '상관 없음'] for member in group])
        has_leader = 1 if leaders >= 1 else 0
        
        # 중복자
        for member in group:
            member_tuple = tuple(member.items())
            if member_tuple in all_members:
                duplicated_members.add(member_tuple)
            else:
                all_members.add(member_tuple)
        
        # 적합도 점수 계산
        total_score += (
            weight_age_std * (group_size - age_std)
            - weight_gender_balance * gender_balance
            - weight_drinking * drinking_issue
            + weight_leader * has_leader)
            
        # 중복된 인원에 대한 패널티 추가
        penalty = len(duplicated_members) * 100000
        total_score -= penalty


    return total_score


# 선택 함수
def selection_function(population):
    selected_population = []
    population_size = len(population)
    for _ in range(population_size):
        # 두 개의 해 선택
        candidate1, candidate2 = random.sample(population, 2)

        # 선택된 두 개의 해 중 더 나은 해 선택
        if fitness_function(candidate1) > fitness_function(candidate2):
            selected_population.append(candidate1)
        else:
            selected_population.append(candidate2)
    return selected_population


# 교차 함수
def crossover_function(selected_population):
    offspring_population = []
    for _ in range(len(selected_population) // 2):
        parent1 = random.choice(selected_population)
        parent2 = random.choice(selected_population)

        crossover_point = random.randint(1, len(parent1) - 1)
        offspring1 = {**dict(list(parent1.items())[:crossover_point]), **dict(list(parent2.items())[crossover_point:])}
        offspring2 = {**dict(list(parent2.items())[:crossover_point]), **dict(list(parent1.items())[crossover_point:])}

        offspring_population.extend([offspring1, offspring2])

    return offspring_population

# 변이 함수
def mutation_function(offspring_population, mutation_rate):
    mutated_population = []
    for offspring in offspring_population:
        for group_number, group in offspring.items():
            if len(group) < 2:
                continue

            if random.random() < mutation_rate:
                idx1, idx2 = random.sample(range(len(group)), 2)
                group[idx1], group[idx2] = group[idx2], group[idx1]
        mutated_population.append(offspring)
    return mutated_population

# 유전 알고리즘 함수
def genetic_algorithm(df, population_size = 50, generations=50, mutation_rate=0.2):
    population = create_initial_population(df, population_size)
    
    for gen in range(generations):
        selected_population = selection_function(population)
        offspring_population = crossover_function(selected_population)
        population = mutation_function(offspring_population, mutation_rate)
        
        # 최고의 적합도 점수 추적
        best_solution = max(population, key=fitness_function)
        best_fitness = fitness_function(best_solution)
        
        # 진척도 출력
        print(f"Generation {gen+1}/{generations}: Best fitness = {best_fitness}")
        
    return best_solution

def create_solution_dataframe(solution):

    # 데이터를 DataFrame으로 변환합니다.
    df = pd.DataFrame()

    for group_num, group_data in solution.items():
        group_df = pd.DataFrame(group_data)
        group_df['조'] = group_num
        df = pd.concat([df, group_df])

    # 결과를 출력합니다.
    return df

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        population_size = int(request.form['population_size'])
        generations = int(request.form['generations'])
        mutation_rate = float(request.form['mutation_rate'])

        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        df = pd.read_csv(filepath)
        df = preprocess_data(df)
        solution = genetic_algorithm(df, population_size, generations, mutation_rate)

        # 결과를 csv 파일로 저장하거나, 필요한 경우 다른 형식으로 저장합니다.
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
        solution_df = create_solution_dataframe(solution)
        solution_df.to_csv(result_path, index=False)

        return redirect(url_for('result', result_file=f'result_{filename}'))

    return render_template('index.html')

def send_from_directory(directory, filename, **options):
    """Send a file from within a directory using send_file().
    ...
    """
    return send_file(os.path.join(directory, filename), **options)


@app.route('/result/<result_file>')
def result(result_file):
    return render_template('result.html', result_file=result_file)

@app.route('/download/<result_file>')
def download(result_file):
    return send_from_directory(app.config['UPLOAD_FOLDER'], result_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
