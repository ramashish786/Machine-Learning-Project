
### INTRODUCTION

Subscription Products often are the main source of revenue for companies across all industries. These
products can come in the form of a 'one size fits all' over compassing subscription, or in multi-level
memberships. Regardless of how they structure their memberships, or what industry they are in,
companies almost always try to minimize customer churn (a.k.a. subscription cancellations). To retain their
customers, these companies first need to identify behavioral patterns that act as catalyst in
disengagement with the product.

* Market: The target audience is the entirety of a company's subscription base. They are the ones
companies want to keep.

* Product: The subscription products that customers are already enrolled in can provide value that users
may not have imagined, or that they may have forgotten.

* Goal: The objective of this model is to predict which users are likely to churn, so that the company can
focus on re-engaging these users with the product. These efforts can be email reminders about the
benefits of the product, especially focusing on features that are new or that the user has shown to value.


### BUSINESS CHALLENGE

* In this Case Study we will be working for a fintech company that provides a subscription product to its users, which allows
them to manage their bank accounts (saving accounts, credit cards, etc), provide them with personalized coupons, inform them of the latest low-APR
loans available in the market, and educates them on the best available methods to save money (like videos on saving money on taxes, free courses on financial health, etc).

* We are in charge of identifying users who are likely to cancel their subscriptions so that we can start building new
features that they may be interested in. These features can increase the engagement and interest of our users.

### DATA

* By subscribing to the membership, our customers have provided us with data on their finances, as well as how they handle those finances through the product. We also have some demographic information we acquired from them during the sign-up process.

* Financial data can often be unreliable and delayed. As a result, companies can sometimes build their marketing models using only demographic data, and data related to finances handled through the product itself. Therefore, we will be restricting ourselves to only using that type of
data. Furthermore, product-related data is more indicative of what new features we should be creating as a company.


Description of each Columns
userid - MongoDB userid
churn  - Active = No | Suspended < 30 = No Else Churn = Yes
age - age of the customer
city - city of the customer
state- state where the customer lives
postal_code - zip code of the customer
zodiac_sign- zodiac sign of the customer
rent_or_own - Does the customer rents or owns a house 
more_than_one_mobile_device - does the customer use more than one mobile device
payFreq- Pay Frequency of the cusomter
in_collections - is the customer in collections
loan_pending - is the loan pending
withdrawn_application - has the customer withdrawn the loan applicaiton 
paid_off_loan- has the customer paid of the loan
did_not_accept_funding - customer did not accept funding
cash_back_engagement - Sum of cash back dollars received by a customer / No of days in the app
cash_back_amount - Sum of cash back dollars received by a customer
used_ios- Has the user used an iphone
used_android - Has the user used a android based phone
has_used_mobile_and_web - Has the user used mobile and web platforms
has_used_web - Has the user used MoneyLion Web app
has_used_mobile - as the user used MoneyLion  app
has_reffered- Has the user referred
cards_clicked - How many times a user has clicked the cards
cards_not_helpful- How helpful was the cards
cards_helpful- How helpful was the cards
cards_viewed- How many times a user viewed the cards
cards_share- How many times a user shared his cards
trivia_view_results-How many times a user viewed trivia results
trivia_view_unlocked- How many times a user viewed trivia view unlocked screen
trivia_view_locked - How many times a user viewed trivia view locked screen
trivia_shared_results- How many times a user shared trivia results 
trivia_played - How many times a user played trivia 
re_linked_account- Has the user re linked account
un_linked_account - Has the user un linked account
credit_score - Customer's credit score

### CONCLUSION

* Our model has provided us with an indication of which users are likely to churn. We have purposefully
left the date of the expected churn open-ended because we are focused on only gauging the features that indicate disengagement
with the product, and not the exact manner (like timeframe) in which users will disengage. In this case study we have chosen this open-ended emphasis to get a sense of those who are even just a bit likely to churn because we are not aiming to create new products for people who are going to leave us for sure, but for people who are starting to lose interest in the app. If, after creating new product features, we start seeing our model predict that fewer of our users are
going to churn, then we can assume our customers are feeling more engaged with what we are offering them. We can move forward with these efforts by inquiring the opinions of our users about our new features (eg. polls). If we want to transition into predicting churn more accurately, in order to
put emphasis strictly on those leaving us, then we can add a time dimension to churn, which would
add more accuracy to our model.
