---
layout: default
---

<!-- What is the central question or problem this project explores? -->
The COVID-19 pandemic had a profound impact on small businesses and restaurants across the globe. The restrictions imposed by local and state governments in an effort to curb the spread of the virus forced businesses to alter their strategies. Many struggled with this change, but some were able to overcome it and remain profitable.  

<!-- Why is this topic relevant or important to understand? -->
For this project, we will be exploring the effects of the COVID-19 pandemic on restaurants and small businesses in the cities of Tampa and Philadelphia. Understanding the factors that enabled certain businesses to thrive during the pandemic can provide valuable insights into consumer behavior, and effective strategies for future disruptions. These findings could help policymakers and restaurant owners make data driven decisions to optimize operations, and allocate resources more effectively in the face of economic shocks.

<!-- Where does the data come from? -->
<!-- What does the dataset cover (scope, time period, granularity) and why? (show some time based plots) -->
We will be using the Yelp Open Dataset, a subset of data that Yelp has collected on over 150,000 businesses across 11 different metro areas, including millions of reviews. This project will focus on two cities, Tampa and Philadelphia, as they have the most data available. While the dataset includes information and reviews on a variety of businesses, the analysis will be focused on restaurants. Furthermore, the timeframe of interest spans from 2019, the year before the pandemic began, through the end of 2020, capturing both baseline activity and the early impact of COVID-19.

<!-- Why These Two Cities -->
There were different approaches when COVID struck, on the one hand, we have cities that were very strict with their local regulations, like Philadelphia; on the other, there were those that were more lax, such as Tampa.

<!-- why only restaurants  impact of covid on restaurant (use external source) and show calplot -->
One of the easiest ways of visualizing the impact of it without going to the extreme cases is with hospitality businesses, more particularly, restaurants. Restaurants were affected by the lockdown conditions, but some of them were still open and changed their tactics to still provide to their clientele.

<!-- explain data analyst methods -->


<!-- how did we label closed and open due to covid -->
To measure the success of restaurants, we need to identify those that closed due to COVID. This is done by filtering for restaurants and further filtering for those that received reviews during the COVID period. Additionally, restaurants marked as "closed" on Yelp are grouped together, as this indicates they were open at some point (to receive a review) and closed afterward. Those still marked as "open" are grouped into the open category.  
  
We wanted to visualize the restaurants that were open and closed in both cities with the hopes of finding some type of pattern for this general aspect of the project, but we found out that more or less the restaurants that closed, did it rather in a uniform way. This can be seen in the two following maps.
<div style="width: 100%; height: 500px;">
  <iframe src="{{ site.baseurl }}/assets/philadelphia_map.html" frameborder="0" width="100%" height="100%"></iframe>
</div>
<div style="text-align: center; font-size: 0.8em; margin-top: 10px; margin-bottom: 20px;">
  <em>Map of Philadelphia restaurants: Green markers indicate restaurants that remained open through the pandemic, while red markers show restaurants that closed. The distribution appears relatively uniform across the city, suggesting that location within Philadelphia was not a major determining factor in restaurant survival.</em>
</div>

<div style="width: 100%; height: 500px;">
  <iframe src="{{ site.baseurl }}/assets/tampa_map.html" frameborder="0" width="100%" height="100%"></iframe>
</div>
<div style="text-align: center; font-size: 0.8em; margin-top: 10px; margin-bottom: 20px;">
  <em>Map of Tampa restaurants: Green markers show open restaurants while red markers indicate closed establishments. As with Philadelphia, the spatial distribution does not reveal clear geographic patterns in restaurant closures, suggesting that factors other than location played more significant roles in business survival.</em>
</div>

<!-- show attribute significance -->
The data shows clear differences in attributes between open and closed restaurants, with statistically significant patterns. Open restaurants were significantly more likely to offer services like delivery, drive-thru, bike parking, and takeout features that align convenience and accessibility with social distancing, contactless service, and mobility during lockdown. In contrast, closed restaurants more often emphasized in-person experiences such as reservations, group seating, alcohol service, and happy hour that became liabilities under COVID restrictions. Therefore the data suggests that restaurants that offered remote-friendly, low-contact service models were better positioned to endure the pandemic's impact. It is important to remember that while these patterns are statistically significant, they reflect correlations not definitive evidence of causation.


<!-- show difference in reviews for open and closed -->
The word cloud illustrates the most common phrases found in reviews with ratings below 2 stars. On the left, we see reviews for closed restaurants, and on the right, reviews for open ones. This allows us to analyze customer sentiments and opinions, comparing the reasons behind the success of the open restaurants. Common complaints for open restaurants center around customer service, while closed ones highlight issues such as food quality, pricing, and hygiene. This comparison suggests that restaurant owners should prioritize food quality and hygiene (duh) over other factors for the best chance of success. However, correlation is not causation.
[Insert image of word cloud]


The importance of contactless service can be further explored in the following graph. This graph shows the percentage of monthly reviews that mention terms such as "delivery" or "to go".

![Common complaints in Bad Reviews](assets/wordcloudtermfreq.png)

This shows that the spike in "delivery" mentions occurred around the COVID period, indicating increased demand for contactless service. Furthermore, combining this with the fact that open restaurants provided significantly more contactless options than closed ones, we can conclude that one possible reason some restaurants closed is that they may have lacked adequate contactless service. However, correlation does not imply causation.


The importance of contactless service can be further explored in the following graph. This graph shows the percentage of monthly reviews that mention terms such as "delivery" or "to go".

![Percentage of monthly reviews that mention the delivery terms](assets/Takeaway.png)

This shows that the spike in delivery terms mentions occurred around the COVID period, indicating increased demand for contactless service. Furthermore, combining this with the fact that open restaurants provided significantly more contactless options than closed ones, we can conclude that one possible reason some restaurants closed is that they may have lacked adequate contactless service

<!-- delivery apps part -->

Mentions of delivery apps reveal a similar story. We can see the sharp increase in the mention of third party delivery apps like Grubhub, DoorDash, and UberEats with the beginning of the pandemic

![Common complaints in Bad Reviews](assets/deliveryapps.png)

The marked increase in mentions across nearly all major platforms reflects how deeply embedded these services became in everyday dining habits. Although many restaurants may offer delivery services of their own, users seem to prefer the convenience, ease of use, and familiarity with larger mainstream apps and services, as opposed to delivery options that may be specific to a particular restaurant. Many restaurants that did not already offer delivery services seem to have taken advantage of the new avenue of business that was created by these apps during the pandemic. For many, partnerships with delivery apps were not just convenient, but essential for survival.


<!-- show cuisine impacts using bokeh -->
To better understand how different types of cuisines were affected by the pandemic, we visualized monthly review counts for the ten most popular cuisine categories using interactive Bokeh plots. The first plot presents the absolute number of reviews over time, revealing overall volume and seasonal patterns. The second plot normalizes each cuisine's review counts relative to the level they had previous to the pandemic, making it easier to compare how much each cuisine was impacted.
<div style="width: 100%; height: 500px;">
  <iframe src="{{ site.baseurl }}/assets/monthly_reviews_cuisine.html" frameborder="0" width="100%" height="100%"></iframe>
</div>

<div style="width: 100%; height: 500px;">
  <iframe src="{{ site.baseurl }}/assets/monthly_reviews_cuisine_normalized.html" frameborder="0" width="100%" height="100%"></iframe>
</div>

<!-- show locations of open and closed restaurants -->




<!-- some conclusion -->

