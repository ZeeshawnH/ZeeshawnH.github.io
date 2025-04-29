#%%
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import tqdm
import ast
import scipy.cluster.hierarchy as sch
literal_eval = ast.literal_eval

"""
- map of closed locations and open ones to compare
- try review count heat map
- bokeh category/ attribute bar interactive 
- Time plot for restrictions 
- calender plot for review counts for all restaurant
- Word cloud for reviews for open and closed restaurants
- 
"""

#%%
# load the businesses data and show basic info
BUSINESSES = pd.read_json("data/yelp_academic_dataset_business.json", lines=True, engine="pyarrow")
print(BUSINESSES.columns)
print("number of citys: ", BUSINESSES['city'].nunique())
print("number of states: ", BUSINESSES['state'].nunique())
print("number of categories: ", BUSINESSES['categories'].nunique())

freq_city = BUSINESSES['city'].value_counts()
print("mean number of business in a city: ", freq_city.mean())
print("median number of business in a city: ", freq_city.median())
print("max number of business in a city: ", freq_city.max(), "city: ", freq_city.idxmax())
print("number of cities with more  than 1000 business: ", freq_city[freq_city > 1000].count())
print("the cities with more than 1000: ", freq_city[freq_city > 1000].index.tolist())

#%% plots
plt.figure(figsize=(15, 10));
state_name_map = {
    'AK': 'Alaska',        'AL': 'Alabama',       'AR': 'Arkansas',      'AS': 'American Samoa', 'AZ': 'Arizona',       'CA': 'California',
    'CO': 'Colorado',      'CT': 'Connecticut',   'DC': 'District of Columbia', 'DE': 'Delaware', 'FL': 'Florida',      'FM': 'Federated States of Micronesia',
    'GA': 'Georgia',       'GU': 'Guam',          'HI': 'Hawaii',        'IA': 'Iowa',           'ID': 'Idaho',         'IL': 'Illinois',
    'IN': 'Indiana',       'KS': 'Kansas',        'KY': 'Kentucky',      'LA': 'Louisiana',      'MA': 'Massachusetts', 'MD': 'Maryland',
    'ME': 'Maine',         'MH': 'Marshall Islands', 'MI': 'Michigan',   'MN': 'Minnesota',      'MO': 'Missouri',      'MP': 'Northern Mariana Islands',
    'MS': 'Mississippi',   'MT': 'Montana',       'NC': 'North Carolina', 'ND': 'North Dakota',  'NE': 'Nebraska',      'NH': 'New Hampshire',
    'NJ': 'New Jersey',    'NM': 'New Mexico',    'NV': 'Nevada',        'NY': 'New York',       'OH': 'Ohio',          'OK': 'Oklahoma',
    'OR': 'Oregon',        'PA': 'Pennsylvania',  'PR': 'Puerto Rico',   'PW': 'Palau',          'RI': 'Rhode Island',  'SC': 'South Carolina',
    'SD': 'South Dakota',  'TN': 'Tennessee',     'TX': 'Texas',         'UT': 'Utah',           'VA': 'Virginia',      'VI': 'Virgin Islands',
    'VT': 'Vermont',       'WA': 'Washington',    'WI': 'Wisconsin',     'WV': 'West Virginia',  'WY': 'Wyoming', 
    "AB": 'Alberta'
}
freq_state = BUSINESSES['state'].value_counts()

mapped_states = [state_name_map.get(state, state) for state in freq_state.index]
plt.bar(mapped_states, freq_state.values)
plt.xticks(rotation=90);
print("business with nan state: ", BUSINESSES[BUSINESSES['state'].isna()].shape[0])
plt.title("Number of businesses in each state")
plt.xlabel("State")
plt.ylabel("Number of businesses")



plt.figure(figsize=(15, 10));
plt.bar(freq_city.index[freq_city > 1000], freq_city.values[freq_city > 1000])
plt.xticks(rotation=90);
plt.title("Cities with more than 1000 businesses")
plt.xlabel("City")
plt.ylabel("Number of businesses")

#%%
# Category information and plot
cat_freq =dict()


def get_categories(categories):
    categories = [cat.strip() for cat in categories.strip().split(',')]
    quoted_cats = [f'"{cat}"' for cat in categories]
    quoted_cats = '[' + ', '.join(quoted_cats) + ']'
    return quoted_cats



BUSINESSES = BUSINESSES.dropna(subset=['categories']).copy()
BUSINESSES['categories'] = BUSINESSES['categories'].apply(get_categories).apply(literal_eval)

cat_freq = {}
for ls in BUSINESSES['categories']:
    for cat in ls:
        cat_freq[cat] = cat_freq.get(cat, 0) + 1
        


cat_freq = dict(sorted(cat_freq.items(), key=lambda item: item[1], reverse=True))
print("number of categories: ", len(cat_freq))
print("mean number of business in a category: ", np.mean(list(cat_freq.values())))
print("median number of business in a category: ", np.median(list(cat_freq.values())))
print("max number of business in a category: ", np.max(list(cat_freq.values())))
print("max category: ", list(cat_freq.keys())[np.argmax(list(cat_freq.values()))])
print("number of categories with more than 1000 business: ", len([i for i in cat_freq.values() if i > 1000]))
n = 50
print(f"the top {n} categories: ", list(cat_freq.keys())[:n])

cat_freq_top50 = dict(list(cat_freq.items())[:50])
plt.figure(figsize=(30, 10));
plt.bar(cat_freq_top50.keys(), cat_freq_top50.values())
plt.xticks(rotation=90);
plt.title("Top 50 categories")
plt.xlabel("Category")
plt.ylabel("Number of businesses")


# %%
# Dealing with attributes 
attribute_ls = list(BUSINESSES.iloc[0]['attributes'].keys())
attribute_freq = dict.fromkeys(attribute_ls, 0)
attribute_df = pd.DataFrame(columns=["business_id"]+attribute_ls)

rows = []

for _, row in tqdm.tqdm(BUSINESSES.iterrows(), total=BUSINESSES.shape[0]):
    id = row['business_id']
    attributes = row['attributes']
    
    new_row = [id] + [None] * len(attribute_ls)
    if isinstance(attributes, dict):
        for i, attr in enumerate(attribute_ls):
            value = attributes.get(attr)
            if value is not None:
                attribute_freq[attr] += 1
                try:
                    new_row[i + 1] = literal_eval(value)
                except:
                    new_row[i + 1] = value
                    print(f"Failed to parse {attr}: {value}, type: {type(value)}")

rows.append(new_row)

attribute_df = pd.DataFrame(rows, columns=["business_id"] + attribute_ls)
attribute_freq = dict(sorted(attribute_freq.items(), key=lambda item: item[1], reverse=True))


print("number of attributes: ", len(attribute_freq))
print("mean number of business with an attribute: ", np.mean(list(attribute_freq.values())))


plt.figure(figsize=(20, 10));
plt.bar(attribute_freq.keys(), attribute_freq.values())
plt.xticks(rotation=90);
plt.title("Number of businesses with each attribute")
plt.xlabel("Attribute")
plt.ylabel("Number of businesses")
# %%
# Dividing by city and loading city, checkin, review data
red_city = "Tampa"
blue_city = "Philadelphia"

businesses = BUSINESSES.copy()
businesses = businesses[businesses['city'].isin([red_city, blue_city])]



red_business = businesses[businesses['city'] == red_city]['business_id'].tolist()
blue_business = businesses[businesses['city'] == blue_city]['business_id'].tolist()



review_file = 'data/yelp_academic_dataset_review.json'
restaurant_business_ids =  businesses[businesses['city'].isin([red_city, blue_city])]['business_id'].tolist()

try:
    checkins = pd.read_json("data/yelp_academic_dataset_checkin.json", lines=True, engine="pyarrow")
    print(checkins.columns.tolist())
    reviews = pd.read_json("data/yelp_academic_dataset_review.json", lines=True,engine = "pyarrow")
    print(reviews.columns.tolist())

    reviews['date'] = pd.to_datetime(reviews['date'])
    print("latest review date: ", reviews['date'].max())
    print("earliest review date: ", reviews['date'].min())

    if 'business_id' not in reviews.columns:
        raise KeyError("Column 'business_id' not found in review data.")
    

    reviews = reviews[reviews['business_id'].isin((red_business + blue_business))]
    checkins = checkins[checkins['business_id'].isin((red_business + blue_business))]
except Exception as e:
    # Fallback: Load a sample to debug
    reviews = pd.read_json("data/yelp_academic_dataset_review.json", lines=True, nrows=10)
    print("Error loading data:", e)

print("number of reviews: ", len(reviews))
print("number of checkins: ", len(checkins))

reviews.loc[reviews['business_id'].isin(red_business), 'city'] = red_city
reviews.loc[reviews['business_id'].isin(blue_business), 'city'] = blue_city


checkins.loc[checkins['business_id'].isin(red_business), 'city'] = red_city
checkins.loc[checkins['business_id'].isin(blue_business), 'city'] = blue_city   


checkins['date_list'] = None
checkins['date_list'] = checkins['date'].apply(lambda x: [datetime.datetime.strptime(date.strip(), "%Y-%m-%d %H:%M:%S") 
                                                         for date in x.split(',')])

print(businesses['city'].value_counts())



#%% 
# Clustering based on busines "type"
red_popuation = 398_325 
blue_popuation = 1_578_000

business_type = "Restaurants"
businesses_id_type = businesses[businesses['categories'].apply(lambda x: business_type in x if isinstance(x, list) else False)]["business_id"].tolist()





red_num_restaurants = len(red_business)
blue_num_restaurants = len(blue_business)


# Filter data for the years 2018-2021
filtered_data = reviews[(reviews['date'].dt.year >= 2018) & (reviews['date'].dt.year <= 2022)].copy()
filtered_data_bid = filtered_data['business_id'].tolist()

filtered_data = filtered_data[filtered_data['business_id'].isin(businesses_id_type)]


filtered_data['year'] = filtered_data['date'].dt.year
filtered_data['month'] = filtered_data['date'].dt.month
monthly_counts = filtered_data.groupby(['year', 'month', 'city']).size().reset_index(name='review_count')

# Create a datetime column for proper sorting
monthly_counts['date'] = pd.to_datetime(monthly_counts['year'].astype(str) + '-' + monthly_counts['month'].astype(str) + '-01')

# Calculate standardized review counts per 100,000 population
monthly_counts['review_count_per_100k'] = monthly_counts.apply(
    lambda row: (row['review_count'] / red_popuation * 100000) if row['city'] == red_city 
    else (row['review_count'] / blue_popuation * 100000) if row['city'] == blue_city else 0,
    axis=1
)

# Calculate reviews per restaurant
monthly_counts['review_count_per_restaurant'] = monthly_counts.apply(
    lambda row: (row['review_count'] / red_num_restaurants) if row['city'] == red_city 
    else (row['review_count'] / blue_num_restaurants) if row['city'] == blue_city else 0,
    axis=1
)

# Set up the plot with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18), sharex=True)

# Create bar positions
dates = sorted(monthly_counts['date'].unique())
x = np.arange(len(dates))
width = 0.35

# Get data for each city
red_data = monthly_counts[monthly_counts['city'] == red_city]
blue_data = monthly_counts[monthly_counts['city'] == blue_city]

# Create arrays for plotting with 0s for missing months - raw counts
red_counts = [red_data[red_data['date'] == d]['review_count'].values[0] if not red_data[red_data['date'] == d].empty else 0 for d in dates]
blue_counts = [blue_data[blue_data['date'] == d]['review_count'].values[0] if not blue_data[blue_data['date'] == d].empty else 0 for d in dates]

# Create arrays for plotting with 0s for missing months - standardized counts by population
red_counts_std = [red_data[red_data['date'] == d]['review_count_per_100k'].values[0] if not red_data[red_data['date'] == d].empty else 0 for d in dates]
blue_counts_std = [blue_data[blue_data['date'] == d]['review_count_per_100k'].values[0] if not blue_data[blue_data['date'] == d].empty else 0 for d in dates]

# Create arrays for plotting with 0s for missing months - standardized counts by restaurant count
red_counts_per_rest = [red_data[red_data['date'] == d]['review_count_per_restaurant'].values[0] if not red_data[red_data['date'] == d].empty else 0 for d in dates]
blue_counts_per_rest = [blue_data[blue_data['date'] == d]['review_count_per_restaurant'].values[0] if not blue_data[blue_data['date'] == d].empty else 0 for d in dates]

# First subplot: Raw counts
ax1.bar(x - width/2, red_counts, width, label=f"{red_city} (raw)", color='red', alpha=0.7)
ax1.bar(x + width/2, blue_counts, width, label=f"{blue_city} (raw)", color='blue', alpha=0.7)
ax1.set_ylabel('Number of Reviews')
ax1.set_title(f'Monthly Review Counts: {red_city} vs {blue_city} (2018-2021)')
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Second subplot: Standardized counts per 100k population
ax2.bar(x - width/2, red_counts_std, width, label=f"{red_city} (per 100k)", color='indianred', alpha=0.7)
ax2.bar(x + width/2, blue_counts_std, width, label=f"{blue_city} (per 100k)", color='royalblue', alpha=0.7)
ax2.set_ylabel('Reviews per 100k Population')
ax2.set_title(f'Monthly Review Counts per 100,000 Population: {red_city} vs {blue_city} (2018-2021)')
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Third subplot: Reviews per restaurant
ax3.bar(x - width/2, red_counts_per_rest, width, label=f"{red_city} (per restaurant)", color='darkred', alpha=0.7)
ax3.bar(x + width/2, blue_counts_per_rest, width, label=f"{blue_city} (per restaurant)", color='darkblue', alpha=0.7)
ax3.set_xlabel('Month and Year')
ax3.set_ylabel('Reviews per Restaurant')
ax3.set_title(f'Monthly Reviews per Restaurant: {red_city} vs {blue_city} (2018-2021)')
ax3.legend()
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# Customize x-axis labels (show every 3 months to avoid overcrowding)
date_labels = [d.strftime('%b %Y') for d in dates]
ax3.set_xticks(x[::3])
ax3.set_xticklabels(date_labels[::3], rotation=45, ha='right')

# Adjust layout and show
plt.tight_layout()
plt.show()






#%% some prints 

# Print statistics about the data
print(f"Raw data:")
print(f"Total reviews (2018-2021) for {red_city}: {sum(red_counts)}")
print(f"Total reviews (2018-2021) for {blue_city}: {sum(blue_counts)}")
print(f"Average monthly reviews for {red_city}: {np.mean(red_counts):.1f}")
print(f"Average monthly reviews for {blue_city}: {np.mean(blue_counts):.1f}")

print(f"\nStandardized data (per 100,000 population):")
print(f"Total reviews per 100k (2018-2021) for {red_city}: {sum(red_counts) / red_popuation * 100000:.1f}")
print(f"Total reviews per 100k (2018-2021) for {blue_city}: {sum(blue_counts) / blue_popuation * 100000:.1f}")
print(f"Average monthly reviews per 100k for {red_city}: {np.mean(red_counts_std):.1f}")
print(f"Average monthly reviews per 100k for {blue_city}: {np.mean(blue_counts_std):.1f}")

print(f"\nStandardized data (per restaurant):")
print(f"Restaurants in {red_city}: {red_num_restaurants}")
print(f"Restaurants in {blue_city}: {blue_num_restaurants}")
print(f"Average reviews per restaurant for {red_city}: {sum(red_counts) / red_num_restaurants:.1f}")
print(f"Average reviews per restaurant for {blue_city}: {sum(blue_counts) / blue_num_restaurants:.1f}")
print(f"Average monthly reviews per restaurant for {red_city}: {np.mean(red_counts_per_rest):.2f}")
print(f"Average monthly reviews per restaurant for {blue_city}: {np.mean(blue_counts_per_rest):.2f}")


#%% 
# compare type of business in the two cities

cat1 = "Restaurants"
cat2 = "Shopping"

cat1_business = businesses[businesses['categories'].apply(lambda x: cat1 in x if isinstance(x, list) else False)]["business_id"].tolist()
cat2_business = businesses[businesses['categories'].apply(lambda x: cat2 in x if isinstance(x, list) else False)]["business_id"].tolist()


num_cat1 = len(cat1_business)
num_cat2 = len(cat2_business)

# Filter data for the years 2018-2021
filtered_data = reviews[(reviews['date'] >=datetime.datetime(2019, 3, 16) ) & (reviews['date'] <= datetime.datetime(2022, 6, 2))]
filtered_data['year'] = filtered_data['date'].dt.year
filtered_data['month'] = filtered_data['date'].dt.month
filtered_data = filtered_data[filtered_data['business_id'].isin(cat1_business + cat2_business)]
filtered_data["category"] = filtered_data["business_id"].apply(lambda x: cat1 if x in cat1_business else cat2)


# Group by year, month, and city to get review counts
monthly_counts = filtered_data.groupby(['year', 'month', 'category']).size().reset_index(name='review_count')

# Create a datetime column for proper sorting
monthly_counts['date'] = pd.to_datetime(monthly_counts['year'].astype(str) + '-' + monthly_counts['month'].astype(str) + '-01')


# Calculate reviews per restaurant
monthly_counts['review_count_per_bus'] = monthly_counts.apply(
    lambda row: (row['review_count'] / num_cat1) if row['category'] == cat1 
    else (row['review_count'] / num_cat2) if row['category'] == cat2 else 0,
    axis=1
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18), sharex=True)

# Create bar positions
dates = sorted(monthly_counts['date'].unique())
x = np.arange(len(dates))
width = 0.35


# Get data for each city
cat1_data = monthly_counts[monthly_counts['category'] == cat1]
cat2_data = monthly_counts[monthly_counts['category'] == cat2]

# Create arrays for plotting with 0s for missing months - raw counts
cat1_counts = [cat1_data[cat1_data['date'] == d]['review_count'].values[0] if not cat1_data[cat1_data['date'] == d].empty else 0 for d in dates]
cat2_counts = [cat2_data[cat2_data['date'] == d]['review_count'].values[0] if not cat2_data[cat2_data['date'] == d].empty else 0 for d in dates]

ax1.bar(x - width/2, cat1_counts, width, label=f"{cat1} (raw)", color='red', alpha=0.7)
ax1.bar(x + width/2, cat2_counts, width, label=f"{cat2} (raw)", color='blue', alpha=0.7)
ax1.set_ylabel('Number of Reviews')
ax1.set_title(f'Monthly Review Counts: {cat1} vs {cat2} (2018-2021)')
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)


cat1_counts_per_bus = [cat1_data[cat1_data['date'] == d]['review_count_per_bus'].values[0] if not cat1_data[cat1_data['date'] == d].empty else 0 for d in dates]
cat2_counts_per_bus = [cat2_data[cat2_data['date'] == d]['review_count_per_bus'].values[0] if not cat2_data[cat2_data['date'] == d].empty else 0 for d in dates]


# # Third subplot: Reviews per restaurant
ax2.bar(x - width/2, cat1_counts_per_bus, width, label=f"{cat1} (per business)", color='darkred', alpha=0.7)
ax2.bar(x + width/2, cat2_counts_per_bus, width, label=f"{cat2} (per business)", color='darkblue', alpha=0.7)
ax2.set_xlabel('Month and Year')
ax2.set_ylabel('Reviews per Business')
ax2.set_title(f'Monthly Reviews per business for: {red_city} vs {blue_city} (2018-2021)')
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# # Customize x-axis labels (show every 3 months to avoid overcrowding)
date_labels = [d.strftime('%b %Y') for d in dates]
ax2.set_xticks(x[::3])
ax2.set_xticklabels(date_labels[::3], rotation=45, ha='right')

# Adjust layout and show
plt.tight_layout()
plt.show()




#%%
# filtering to resturants and open and revirews in the period
restaurants = businesses[businesses['categories'].apply(lambda x: "Restaurants" in x if isinstance(x, list) else False)]
restaurants_id = restaurants['business_id'].tolist()
restaurants_reviews = reviews[reviews['business_id'].isin(restaurants_id)]
restaurants_reviews = restaurants_reviews[restaurants_reviews['date'] >= datetime.datetime(2019, 1, 1) ]
restaurants = restaurants[restaurants['business_id'].isin(restaurants_reviews['business_id'].tolist())]

restaurants["review_count"] = restaurants_reviews.groupby('business_id').size().reset_index(name='review_count')['review_count']

counts = restaurants['review_count'].tolist()
print("number of restaurants more than 50", sum([1 for i in counts if i > 50]))
print("number of restaurants: ", len(restaurants))
print("number of restaurants with reviews: ", len(restaurants_reviews))
# plt.hist(counts, bins=30, color='blue', alpha=0.7);

#%%
print("closed restaurants: ", restaurants[restaurants['is_open'] == 0].shape[0])
print("closed businesses: ", businesses[businesses['is_open'] == 0].shape[0])

#%%
before_date = datetime.datetime(2020, 12, 30)
after_date = datetime.datetime(2020, 3, 1)
# get all closed restaurants that have reviews in 2019 or 2020
closed_reviews = reviews[(reviews['date'] < before_date) &
                         (reviews['date'] > after_date) &
                          (reviews['business_id'].isin(restaurants_id)) & 
                          (reviews['business_id'].isin(restaurants[restaurants['is_open'] == 0]['business_id'].tolist()))]
closed_restaurants = restaurants[restaurants['business_id'].isin(closed_reviews['business_id'].tolist())]
print("number of closed restaurants: ", len(closed_restaurants))
print("number of reviews for closed restaurants: ", len(closed_reviews))

open_reviews = reviews[(reviews['date'] < before_date) & 
                          (reviews['date'] > after_date) &
                       (reviews['business_id'].isin(restaurants_id)) & 
                       (reviews['business_id'].isin(restaurants[restaurants['is_open'] == 1]['business_id'].tolist()))]
open_restaurants = restaurants[restaurants['business_id'].isin(open_reviews['business_id'].tolist())]
print("number of open restaurants: ", len(open_restaurants))
print("number of reviews for open restaurants: ", len(open_reviews))



# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import re

def create_wordcloud(opendf,closeddf, engram, ratings,words = 20, max_df=0.8):
    closed_text = closeddf[(closeddf['text'].notna())&
                             (closeddf["stars"]<=ratings)]['text'].tolist()
    open_text = opendf[(opendf['text'].notna())&
                                (opendf["stars"]<=ratings)]['text'].tolist()

    closed_text = [text.replace("'", '').replace("-","") for text in closed_text]
    open_text = [text.replace("'", '').replace("-","") for text in open_text]
    # replace numbers with x
    closed_text = [re.sub(r'\d', 'x', text) for text in closed_text]
    open_text = [re.sub(r'\d', 'x', text) for text in open_text]

    vectorizer_closed = TfidfVectorizer(stop_words='english', max_features=10000, lowercase=True, ngram_range=engram, max_df=max_df)
    tfidf_matrix_closed = vectorizer_closed.fit_transform(closed_text )
    tf_idf_closed = pd.DataFrame(tfidf_matrix_closed.toarray(), columns=vectorizer_closed.get_feature_names_out())
    tfidf_sum_closed = tf_idf_closed.sum(axis=0)/len(closed_text)
    sorted_tfidf_closed = tfidf_sum_closed.sort_values(ascending=False)

    vectorizer_open = TfidfVectorizer(stop_words='english', max_features=10000, lowercase=True, ngram_range=engram, max_df=max_df)
    tfidf_matrix_open = vectorizer_open.fit_transform(open_text)
    tf_idf_open = pd.DataFrame(tfidf_matrix_open.toarray(), columns=vectorizer_open.get_feature_names_out())
    tfidf_sum_open = tf_idf_open.sum(axis=0)
    sorted_tfidf_open = tfidf_sum_open.sort_values(ascending=False)/len(open_text)

    tfidf =  sorted_tfidf_closed - sorted_tfidf_open
    tfidf = tfidf[tfidf > 0].sort_values(ascending=False)


    exclusive_closed_words = sorted_tfidf_closed.loc[~sorted_tfidf_closed.index.isin(sorted_tfidf_open.index)]

    # sort descending
    exclusive_closed_words = exclusive_closed_words.sort_values(ascending=False)
    word_freq_open = dict(zip(sorted_tfidf_open.index, sorted_tfidf_open.values))
    word_freq_closed = dict(zip(sorted_tfidf_closed.index, sorted_tfidf_closed.values))
    word_freq_diff = dict(zip(tfidf.index, tfidf.values))

    # combine the two dictionaries
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, lowercase=True, ngram_range=engram, max_df=max_df)
    tfidf_matrix = vectorizer.fit_transform(open_text+closed_text)
    tf_idf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_sum = tf_idf.sum(axis=0)
    sorted_tfidf_both = tfidf_sum.sort_values(ascending=False)/len(open_text)



    wordcloud_closed = WordCloud(
        width=800, 
        height=400,
        max_words=words,
        background_color='white'
    ).generate_from_frequencies(word_freq_closed)

    wordcloud_open = WordCloud(
        width=800, 
        height=400,
        max_words=words,
        background_color='white'
    ).generate_from_frequencies(word_freq_open)

    wordcloud_diff = WordCloud(
        width=800, 
        height=400,
        max_words=words,
        background_color='white'
    ).generate_from_frequencies(sorted_tfidf_both)

    fig, ax = plt.subplots(1, 3, figsize=(20, 30))
    ax[0].imshow(wordcloud_closed, interpolation='bilinear')
    ax[0].set_title("Word Cloud for Closed Restaurants")
    ax[0].axis("off")

    ax[1].imshow(wordcloud_open, interpolation='bilinear')
    ax[1].set_title("Word Cloud for Open Restaurants")
    ax[1].axis("off")

    ax[2].imshow(wordcloud_diff, interpolation='bilinear')
    ax[2].set_title("Word Cloud for Difference")
    ax[2].axis("off")
    plt.tight_layout()




# %%
create_wordcloud(open_reviews, closed_reviews, (3,3), 2, words = 25, max_df= 0.3)
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import re

def create_wordcloud(opendf, closeddf, engram, ratings, words=20, max_df=0.8):
    # group reviews by business_id
    closed_grouped = closeddf[(closeddf['text'].notna()) & (closeddf["stars"] <= ratings)]
    closed_text = closed_grouped.groupby('business_id')['text'].apply(lambda x: ' '.join(x)).tolist()

    open_grouped = opendf[(opendf['text'].notna()) & (opendf["stars"] <= ratings)]
    open_text = open_grouped.groupby('business_id')['text'].apply(lambda x: ' '.join(x)).tolist()

    # clean text
    def clean_text(texts):
        texts = [t.replace("'", '').replace("-", "") for t in texts]
        texts = [re.sub(r'\d+', 'x', t) for t in texts]
        return texts

    closed_text = clean_text(closed_text)
    open_text = clean_text(open_text)

    vectorizer_closed = TfidfVectorizer(stop_words='english', max_features=10000, lowercase=True,
                                        ngram_range=engram, max_df=max_df)
    tfidf_matrix_closed = vectorizer_closed.fit_transform(closed_text)
    tf_idf_closed = pd.DataFrame(tfidf_matrix_closed.toarray(), columns=vectorizer_closed.get_feature_names_out())
    sorted_tfidf_closed = tf_idf_closed.mean(axis=0).sort_values(ascending=False)

    vectorizer_open = TfidfVectorizer(stop_words='english', max_features=10000, lowercase=True,
                                      ngram_range=engram, max_df=max_df)
    tfidf_matrix_open = vectorizer_open.fit_transform(open_text)
    tf_idf_open = pd.DataFrame(tfidf_matrix_open.toarray(), columns=vectorizer_open.get_feature_names_out())
    sorted_tfidf_open = tf_idf_open.mean(axis=0).sort_values(ascending=False)

    tfidf_diff = (sorted_tfidf_closed - sorted_tfidf_open).dropna()
    tfidf_diff = tfidf_diff[tfidf_diff > 0].sort_values(ascending=False)

    exclusive_closed = sorted_tfidf_closed.loc[~sorted_tfidf_closed.index.isin(sorted_tfidf_open.index)].sort_values(ascending=False)

    tfidf_matrix_both = vectorizer_closed.fit_transform(closed_text + open_text)
    tf_idf_both = pd.DataFrame(tfidf_matrix_both.toarray(), columns=vectorizer_closed.get_feature_names_out())
    sorted_tfidf_both = tf_idf_both.mean(axis=0).sort_values(ascending=False)


    wordcloud_closed = WordCloud(width=800, height=400, max_words=words, background_color='white').generate_from_frequencies(sorted_tfidf_closed)
    wordcloud_open = WordCloud(width=800, height=400, max_words=words, background_color='white').generate_from_frequencies(sorted_tfidf_open)
    wordcloud_diff = WordCloud(width=800, height=400, max_words=words, background_color='white').generate_from_frequencies(sorted_tfidf_both)

    fig, ax = plt.subplots(1, 3, figsize=(20, 30))
    ax[0].imshow(wordcloud_closed, interpolation='bilinear')
    ax[0].set_title("Word Cloud for Closed Restaurants")
    ax[0].axis("off")

    ax[1].imshow(wordcloud_open, interpolation='bilinear')
    ax[1].set_title("Word Cloud for Open Restaurants")
    ax[1].axis("off")

    ax[2].imshow(wordcloud_diff, interpolation='bilinear')
    ax[2].set_title("Exclusive Closed Words")
    ax[2].axis("off")
    plt.tight_layout()
# %%
create_wordcloud(open_reviews, closed_reviews, (3,3), 2, words = 25, max_df= 0.3)
# %%
def create_wordcloud_group(opendf, closeddf, engram=(1,2), ratings=2, words=20, max_df=0.8):
    # filter and merge all reviews into one string per group
    closed_text = closeddf[(closeddf['text'].notna()) & (closeddf["stars"] <= ratings)]['text']
    open_text = opendf[(opendf['text'].notna()) & (opendf["stars"] <= ratings)]['text']

    def clean(text_series):
        text = ' '.join(text_series)
        text = text.replace("'", '').replace("-", '')
        text = re.sub(r'\d+', 'x', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return [text]

    closed_text = clean(closed_text)
    open_text = clean(open_text)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, lowercase=True, 
                                 ngram_range=engram, max_df=max_df)

    tfidf = vectorizer.fit_transform(closed_text + open_text)
    feature_names = vectorizer.get_feature_names_out()
    
    tfidf_closed = pd.Series(tfidf.toarray()[0], index=feature_names)
    tfidf_open = pd.Series(tfidf.toarray()[1], index=feature_names)

    tfidf_diff = (tfidf_closed - tfidf_open).sort_values(ascending=False)
    tfidf_diff = tfidf_diff[tfidf_diff > 0]

    tfidf_combined = pd.concat([tfidf_closed, tfidf_open], axis=1)
    
    wordcloud_closed = WordCloud(width=800, height=400, max_words=words, background_color='white').generate_from_frequencies(tfidf_closed)
    wordcloud_open = WordCloud(width=800, height=400, max_words=words, background_color='white').generate_from_frequencies(tfidf_open)
    wordcloud_diff = WordCloud(width=800, height=400, max_words=words, background_color='white').generate_from_frequencies(tfidf_combined)

    fig, ax = plt.subplots(1, 3, figsize=(20, 30))
    ax[0].imshow(wordcloud_closed, interpolation='bilinear')
    ax[0].set_title("Closed Group")
    ax[0].axis("off")

    ax[1].imshow(wordcloud_open, interpolation='bilinear')
    ax[1].set_title("Open Group")
    ax[1].axis("off")

    ax[2].imshow(wordcloud_diff, interpolation='bilinear')
    ax[2].set_title("Exclusive to Closed")
    ax[2].axis("off")
    plt.tight_layout()
# %%
create_wordcloud(open_reviews, closed_reviews, (3,3), 2, words = 25, max_df= 0.3)
# %%