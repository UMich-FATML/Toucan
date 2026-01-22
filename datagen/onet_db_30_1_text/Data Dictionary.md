# O*NET 30.1 Data Dictionary

## Knowledge, Skills, Abilities

### Knowledge

Purpose:	Provide a mapping of O*NET-SOC codes (occupations) to Knowledge ratings.
Download:	Knowledge.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Data Value	Float(5,2)	Rating associated with the O*NET-SOC occupation
N	Integer(4)	Sample size
Standard Error	Float(7,4)	Standard Error
Lower CI Bound	Float(7,4)	Lower 95% confidence interval bound
Upper CI Bound	Float(7,4)	Upper 95% confidence interval bound
Recommend Suppress	Character(1)	Low precision indicator (Y=yes, N=no)
Not Relevant	Character(1)	Not relevant for the occupation (Y=yes, N=no)
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains the Content Model Knowledge data associated with each O*NET-SOC occupation. It is displayed in 13 tab delimited fields and identified using the column names provided above. Item rating level metadata is provided in columns named N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, Not Relevant, Date, and Domain Source. Refer to Appendix 2, Item Rating Level Statistics - Incumbent for additional information on these items. The 13 fields are represented by one row. There are a total of 59,004 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
5.0	Date and Source columns added
5.1	Columns added for N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, and Not Relevant
6.0 - 28.1	No structure changes
28.2	Standard Error, Lower CI Bound, Upper CI Bound expanded from 2 decimal places to 4
28.3 - 30.1	No structure changes

Data Example - Knowledge:
O*NET-SOC Code	Element ID	Element Name	Scale ID	Data Value	N	Standard Error	Lower CI Bound	Upper CI Bound	Recommend Suppress	Not Relevant	Date	Domain Source
29-2011.00	2.C.8.b	Law and Government	IM	2.52	28	0.2879	1.9275	3.1090	N	n/a	08/2025	Incumbent
29-2011.00	2.C.8.b	Law and Government	LV	2.47	28	0.4408	1.5705	3.3792	N	N	08/2025	Incumbent
29-2011.00	2.C.9.a	Telecommunications	IM	2.30	28	0.1504	1.9912	2.6086	N	n/a	08/2025	Incumbent
29-2011.00	2.C.9.a	Telecommunications	LV	1.64	28	0.3761	0.8672	2.4105	N	N	08/2025	Incumbent
29-2011.00	2.C.9.b	Communications and Media	IM	1.80	28	0.2181	1.3530	2.2482	N	n/a	08/2025	Incumbent
29-2011.00	2.C.9.b	Communications and Media	LV	1.46	28	0.4430	0.5467	2.3648	N	N	08/2025	Incumbent
29-2011.00	2.C.10	Transportation	IM	1.75	27	0.1723	1.3918	2.1002	N	n/a	08/2025	Incumbent
29-2011.00	2.C.10	Transportation	LV	1.28	27	0.2805	0.7064	1.8596	N	N	08/2025	Incumbent

### Skills

Purpose:	Provide a mapping of O*NET-SOC codes (occupations) to Skill ratings.
Download:	Skills.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Data Value	Float(5,2)	Rating associated with the O*NET-SOC occupation
N	Integer(4)	Sample size
Standard Error	Float(7,4)	Standard Error
Lower CI Bound	Float(7,4)	Lower 95% confidence interval bound
Upper CI Bound	Float(7,4)	Upper 95% confidence interval bound
Recommend Suppress	Character(1)	Low precision indicator (Y=yes, N=no)
Not Relevant	Character(1)	Not relevant for the occupation (Y=yes, N=no)
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains the Content Model Skill data associated with each O*NET-SOC occupation. It is displayed in 13 tab delimited fields and identified using the column names provided above. Item rating level metadata is provided in columns named N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, Not Relevant, Date, and Domain Source. Refer to Appendix 1, Item Rating Level Statistics - Analyst for additional information on these items. The 13 fields are represented by one row. There are a total of 62,580 rows of data in this file.

For more information, see:

O*NET Analyst Occupational Skills Ratings: Procedures Update

File Structure Changes:
Release Number	Description of Change
5.0	Date and Source columns added
5.1	Columns added for N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, and Not Relevant
6.0 - 28.1	No structure changes
28.2	Standard Error, Lower CI Bound, Upper CI Bound expanded from 2 decimal places to 4
28.3 - 30.1	No structure changes

Data Example - Skills:
O*NET-SOC Code	Element ID	Element Name	Scale ID	Data Value	N	Standard Error	Lower CI Bound	Upper CI Bound	Recommend Suppress	Not Relevant	Date	Domain Source
49-3041.00	2.A.1.a	Reading Comprehension	IM	3.00	8	0.0000	3.0000	3.0000	N	n/a	08/2025	Analyst
49-3041.00	2.A.1.a	Reading Comprehension	LV	2.88	8	0.1250	2.6300	3.1200	N	N	08/2025	Analyst
49-3041.00	2.A.1.b	Active Listening	IM	3.12	8	0.1250	2.8800	3.3700	N	n/a	08/2025	Analyst
49-3041.00	2.A.1.b	Active Listening	LV	3.00	8	0.0000	3.0000	3.0000	N	N	08/2025	Analyst
49-3041.00	2.A.1.c	Writing	IM	2.88	8	0.1250	2.6300	3.1200	N	n/a	08/2025	Analyst
49-3041.00	2.A.1.c	Writing	LV	2.62	8	0.1830	2.2664	2.9836	N	N	08/2025	Analyst
49-3041.00	2.A.1.d	Speaking	IM	3.12	8	0.1250	2.8800	3.3700	N	n/a	08/2025	Analyst
49-3041.00	2.A.1.d	Speaking	LV	3.00	8	0.0000	3.0000	3.0000	N	N	08/2025	Analyst

### Abilities

Purpose:	Provide a mapping of O*NET-SOC codes (occupations) to Ability ratings.
Download:	Abilities.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Data Value	Float(5,2)	Rating associated with the O*NET-SOC occupation
N	Integer(4)	Sample size
Standard Error	Float(7,4)	Standard Error
Lower CI Bound	Float(7,4)	Lower 95% confidence interval bound
Upper CI Bound	Float(7,4)	Upper 95% confidence interval bound
Recommend Suppress	Character(1)	Low precision indicator (Y=yes, N=no)
Not Relevant	Character(1)	Not relevant for the occupation (Y=yes, N=no)
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains the Content Model Ability data associated with each O*NET-SOC occupation. It is displayed in 13 tab delimited fields and identified using the column names provided above. Item rating level metadata is provided in columns named N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, Not Relevant, Date, and Domain Source. Refer to Appendix 1, Item Rating Level Statistics - Analyst for additional information on these items. The 13 fields are represented by one row. There are a total of 92,976 rows of data in this file.

For more information, see:

O*NET Analyst Occupational Ratings: Linkage Revisit
O*NET Analyst Occupational Abilities Ratings: Procedures Update
Updating Occupational Ability Profiles with O*NET Content Model Descriptors
Linking Client Assessment Profiles to O*NET Occupational Profiles Within the O*NET Ability Profiler

File Structure Changes:
Release Number	Description of Change
5.0	Date and Source columns added
5.1	Columns added for N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, and Not Relevant
6.0 - 28.1	No structure changes
28.2	Standard Error, Lower CI Bound, Upper CI Bound expanded from 2 decimal places to 4
28.3 - 30.1	No structure changes

Data Example - Abilities:
O*NET-SOC Code	Element ID	Element Name	Scale ID	Data Value	N	Standard Error	Lower CI Bound	Upper CI Bound	Recommend Suppress	Not Relevant	Date	Domain Source
53-3051.00	1.A.1.a.1	Oral Comprehension	IM	3.25	8	0.1637	2.9292	3.5708	N	n/a	08/2025	Analyst
53-3051.00	1.A.1.a.1	Oral Comprehension	LV	3.38	8	0.1830	3.0164	3.7336	N	N	08/2025	Analyst
53-3051.00	1.A.1.a.2	Written Comprehension	IM	2.75	8	0.1637	2.4292	3.0708	N	n/a	08/2025	Analyst
53-3051.00	1.A.1.a.2	Written Comprehension	LV	3.00	8	0.0000	3.0000	3.0000	N	N	08/2025	Analyst
53-3051.00	1.A.1.a.3	Oral Expression	IM	3.12	8	0.1250	2.8800	3.3700	N	n/a	08/2025	Analyst
53-3051.00	1.A.1.a.3	Oral Expression	LV	3.12	8	0.1250	2.8800	3.3700	N	N	08/2025	Analyst
53-3051.00	1.A.1.a.4	Written Expression	IM	2.75	8	0.1637	2.4292	3.0708	N	n/a	08/2025	Analyst
53-3051.00	1.A.1.a.4	Written Expression	LV	2.75	8	0.1637	2.4292	3.0708	N	N	08/2025	Analyst

## Education, Experience, Training

### Education, Training, and Experience

Purpose:	Provide a mapping of O*NET-SOC codes (occupations) to Education, Training, and Experience ratings.
Download:	Education, Training, and Experience.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Category	Integer(3)	Percent frequency category
Data Value	Float(5,2)	Rating associated with the O*NET-SOC occupation
N	Integer(4)	Sample size
Standard Error	Float(7,4)	Standard Error
Lower CI Bound	Float(7,4)	Lower 95% confidence interval bound
Upper CI Bound	Float(7,4)	Upper 95% confidence interval bound
Recommend Suppress	Character(1)	Low precision indicator (Y=yes, N=no)
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains the percent frequency data associated with Education, Training, and Experience Content Model elements. It is displayed in 13 tab delimited fields and identified using the column names provided above. Item rating level metadata is provided in columns named N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, Date, and Domain Source. Refer to Appendix 2, Item Rating Level Statistics - Incumbent for additional information on these items. The 13 fields are represented by one row. There are a total of 37,125 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
5.0	Initial file introduction
5.1	Columns added for N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress
6.0 - 28.1	No structure changes
28.2	Standard Error, Lower CI Bound, Upper CI Bound expanded from 2 decimal places to 4
28.3 - 30.1	No structure changes

Data Example - Education, Training, and Experience:
O*NET-SOC Code	Element ID	Element Name	Scale ID	Category	Data Value	N	Standard Error	Lower CI Bound	Upper CI Bound	Recommend Suppress	Date	Domain Source
33-9011.00	2.D.1	Required Level of Education	RL	1	0.00	26	n/a	n/a	n/a	n/a	08/2025	Occupational Expert
33-9011.00	2.D.1	Required Level of Education	RL	2	65.38	26	n/a	n/a	n/a	n/a	08/2025	Occupational Expert
33-9011.00	2.D.1	Required Level of Education	RL	3	19.23	26	n/a	n/a	n/a	n/a	08/2025	Occupational Expert
33-9011.00	2.D.1	Required Level of Education	RL	4	0.00	26	n/a	n/a	n/a	n/a	08/2025	Occupational Expert
33-9011.00	2.D.1	Required Level of Education	RL	5	11.54	26	n/a	n/a	n/a	n/a	08/2025	Occupational Expert
33-9011.00	2.D.1	Required Level of Education	RL	6	3.85	26	n/a	n/a	n/a	n/a	08/2025	Occupational Expert
33-9011.00	2.D.1	Required Level of Education	RL	7	0.00	26	n/a	n/a	n/a	n/a	08/2025	Occupational Expert
33-9011.00	2.D.1	Required Level of Education	RL	8	0.00	26	n/a	n/a	n/a	n/a	08/2025	Occupational Expert

### Education, Training, and Experience Categories

Purpose:	Provide descriptions of the Education, Training, and Experience percent frequency categories.
Download:	Education, Training, and Experience Categories.txt
Structure and Description:
Column	Type	Column Content
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Category	Integer(3)	Category value associated with element
Category Description	Character Varying(1000)	Detail description of category

This file contains the categories associated with the Education, Training, and Experience content area. Categories for the following scales are included: Required Level of Education (RL), Related Work Experience (RW), On-Site or In-Plant Training (PT), and On-The-Job Training (OJ). It is displayed in 5 tab delimited fields. There are a total of 41 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
9.0	Added as a new file
10.0 - 30.1	No structure changes

Data Example - Related Work Experience Scale (RW):
Category	Category Description
1	None
2	Up to 1 month
3	Over 1 month, up to 3 months
4	Over 3 months, up to 6 months
5	Over 6 months, up to 1 year
6-11	Progressive increments reaching "Over 10 years"

### Job Zones

Purpose:	Provide a mapping of O*NET-SOC occupations to Job Zone ratings.
Download:	Job Zones.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Job Zone	Integer(1)	Job Zone rating (1-5 scale)
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains each O*NET-SOC code and its corresponding Job Zone number. The file is displayed in four tab delimited fields with the columns named O*NET-SOC Code, Job Zone, Date, and Domain Source. The four fields are represented by one row. There are a total of 923 rows of data in this file.

For more information, see:

Procedures for O*NET Job Zone Assignment
Procedures for O*NET Job Zone Assignment: Updated to Include Procedures for Developing Preliminary Job Zones for New O*NET-SOC Occupations

File Structure Changes:
Release Number	Description of Change
5.0	No changes
5.1	Date and Domain Source columns added
6.0 - 30.1	No structure changes

Data Example - Job Zones:
O*NET-SOC Code	Job Zone	Date	Domain Source
17-3026.01	4	08/2025	Analyst
27-4014.00	3	08/2025	Analyst
45-3031.00	1	08/2025	Analyst

### Job Zone Reference

Purpose:	Provide Job Zone data (developed to help transition DOT's measures of Specific Vocational Preparation (SVP) to O*NET's measure of experience, education, and job training).
Download:	Job Zone Reference.txt
Structure and Description:
Column	Type	Column Content
Job Zone	Integer(1)	Job Zone number
Name	Character Varying(50)	Job Zone name
Experience	Character Varying(300)	Job Zone experience requirements
Education	Character Varying(500)	Job Zone educational requirements
Job Training	Character Varying(300)	Job Zone training requirements
Examples	Character Varying(500)	Job Zone examples
SVP Range	Character Varying(25)	Specific vocational preparation range

This file describes the five O*NET Job Zones, which are groups of occupations that need the same level of experience, education, and training. The file is displayed in seven tab delimited fields with the columns named Job Zone, Name, Experience, Education, Job Training, Examples, and SVP Range. The seven fields are represented by one row. There are a total of 5 rows of data in this file.

For more information, see:

Procedures for O*NET Job Zone Assignment
Procedures for O*NET Job Zone Assignment: Updated to Include Procedures for Developing Preliminary Job Zones for New O*NET-SOC Occupations

File Structure Changes:
Release Number	Description of Change
5.0 - 30.1	No structure changes

Data Example - Job Zone Reference:
Job Zone 1 represents occupations requiring minimal preparation, typically needing "anywhere from a few days to a few months of training." Examples include dishwashers and groundskeeping workers, with SVP Range below 4.0.

## Interests

### Interests

Purpose:	Provide general occupational interest (RIASEC) high-point codes and numeric profile data for each O*NET-SOC occupation.
Download:	Interests.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Data Value	Float(5,2)	Rating associated with the O*NET-SOC occupation
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains the general occupational interest (RIASEC) high-point codes and numeric profile data for each O*NET-SOC occupation. Interest ratings are presented as two scales: OI reports the RIASEC level of each interest and IH presents "high-point codes", the numbers of the RIASEC scales for the first, second and/or third highest ratings. The high-point values represent the following elements:

0.00 = No high point available
1.00 = Realistic
2.00 = Investigative
3.00 = Artistic
4.00 = Social
5.00 = Enterprising
6.00 = Conventional

The file is displayed in seven tab delimited fields with the columns named O*NET-SOC Code, Element ID, Element Name, Scale ID, Data Value, Date, and Domain Source. The seven fields are represented by one row. There are a total of 8,307 rows of data in this file.

For more information, see:

Using Machine Learning to Develop Occupational Interest Profiles and High-Point Codes for the O*NET System
Career Returns within the O*NET Interest Profiler Tools
Development of an O*NET Mini Interest Profiler (Mini-IP) for Mobile Devices: Psychometric Characteristics
Examining the Efficacy of Emoji Anchors for the O*NET Interest Profiler Short Form
O*NET Interest Profiler Short Form Psychometric Characteristics: Summary

File Structure Changes:
Release Number	Description of Change
5.0	Date and Source columns added
5.1 - 30.1	No structure changes

Data Example - Interests:
O*NET-SOC Code	Element ID	Element Name	Scale ID	Data Value	Date	Domain Source
43-4041.00	1.B.1.a	Realistic	OI	1.00	11/2023	Machine Learning
43-4041.00	1.B.1.b	Investigative	OI	1.85	11/2023	Machine Learning
43-4041.00	1.B.1.c	Artistic	OI	1.00	11/2023	Machine Learning
43-4041.00	1.B.1.d	Social	OI	3.39	11/2023	Machine Learning
43-4041.00	1.B.1.e	Enterprising	OI	4.47	11/2023	Machine Learning
43-4041.00	1.B.1.f	Conventional	OI	7.00	11/2023	Machine Learning
43-4041.00	1.B.1.g	First Interest High-Point	IH	6.00	11/2023	Machine Learning
29-2034.00	1.B.1.a	Realistic	OI	6.25	11/2023	Machine Learning

### RIASEC Keywords

Purpose:	Provide action and object keywords for each general occupational interest.
Download:	RIASEC Keywords.txt
Structure and Description:
Column	Type	Column Content
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Keyword	Character Varying(150)	Relevant interest keyword
Keyword Type	Character Varying(20)	"Action" or "Object" designation

This file contains action and object keywords for each general occupational interest. The file is displayed in four tab delimited fields with the columns named Element ID, Element Name, Keyword, and Keyword Type. The four fields are represented by one row. There are a total of 75 rows of data in this file.

For more information, see:

Updating Vocational Interests Information for the O*NET Content Model

File Structure Changes:
Release Number	Description of Change
27.2	Added as a new file
27.3 - 30.1	No structure changes

Data Example - RIASEC Keywords:
Element ID	Element Name	Keyword	Keyword Type
1.B.1.a	Realistic	Build	Action
1.B.1.a	Realistic	Drive	Action
1.B.1.a	Realistic	Install	Action
1.B.1.a	Realistic	Maintain	Action
1.B.1.a	Realistic	Repair	Action

### Basic Interests to RIASEC

Purpose:	Provide linkages between each basic occupational interest to relevant general occupational interests.
Download:	Basic Interests to RIASEC.txt
Structure and Description:
Column	Type	Column Content
Basic Interests Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Basic Interests Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
RIASEC Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
RIASEC Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)

This file contains linkages between each basic occupational interest to relevant general occupational interests. The file is displayed in four tab delimited fields with the columns named Basic Interests Element ID, Basic Interests Element Name, RIASEC Element ID, and RIASEC Element Name. The four fields are represented by one row. There are a total of 53 rows of data in this file.

For more information, see:

Updating Vocational Interests Information for the O*NET Content Model

File Structure Changes:
Release Number	Description of Change
27.2	Added as a new file
27.3 - 30.1	No structure changes

Data Example - Basic Interests to RIASEC:
Basic Interests Element ID	Basic Interests Element Name	RIASEC Element ID	RIASEC Element Name
1.B.3.a	Mechanics/Electronics	1.B.1.a	Realistic
1.B.3.b	Construction/Woodwork	1.B.1.a	Realistic
1.B.3.c	Transportation/Machine Operation	1.B.1.a	Realistic
1.B.3.d	Physical/Manual Labor	1.B.1.a	Realistic
1.B.3.e	Protective Service	1.B.1.a	Realistic

### Interests Illustrative Activities

Purpose:	Provide illustrative work activities related to the general and basic occupational interests.
Download:	Interests Illustrative Activities.txt
Structure and Description:
Column	Type	Column Content
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Interest Type	Character Varying(20)	"General" or "Basic" designation
Activity	Character Varying(150)	Sample work activity

This file contains illustrative work activities related to the general and basic occupational interests. The file is displayed in four tab delimited fields with the columns named Element ID, Element Name, Interest Type, and Activity. The four fields are represented by one row. There are a total of 188 rows of data in this file.

For more information, see:

Updating Vocational Interests Information for the O*NET Content Model

File Structure Changes:
Release Number	Description of Change
27.2	Added as a new file
27.3 - 30.1	No structure changes

Data Example - Interests Illustrative Activities:
Element ID	Element Name	Interest Type	Activity
1.B.1.a	Realistic	General	Build kitchen cabinets.
1.B.1.a	Realistic	General	Drive a truck to deliver packages to offices and homes.
1.B.1.a	Realistic	General	Put out forest fires.
1.B.1.a	Realistic	General	Repair household appliances.
1.B.1.b	Investigative	General	Develop a new medicine.

### Interests Illustrative Occupations

Purpose:	Provide illustrative occupations linked to the general and basic occupational interests.
Download:	Interests Illustrative Occupations.txt
Structure and Description:
Column	Type	Column Content
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Interest Type	Character Varying(20)	"General" or "Basic" designation
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)

This file contains illustrative occupations linked to the general and basic occupational interests. For occupation-specific ratings for RIASEC elements, see the Interests file. The file is displayed in four tab delimited fields with the columns named Element ID, Element Name, Interest Type, and O*NET-SOC Code. The four fields are represented by one row. There are a total of 186 rows of data in this file.

For more information, see:

Updating Vocational Interests Information for the O*NET Content Model

File Structure Changes:
Release Number	Description of Change
27.2	Added as a new file
27.3 - 30.1	No structure changes

Data Example - Interests Illustrative Occupations:
Element ID	Element Name	Interest Type	O*NET-SOC Code
1.B.1.a	Realistic	General	17-3024.01
1.B.1.a	Realistic	General	45-2091.00
1.B.1.a	Realistic	General	47-2031.00
1.B.1.a	Realistic	General	53-3052.00
1.B.1.b	Investigative	General	19-1029.04

## Work Styles

### Work Styles

Purpose:	Provide a mapping of O*NET-SOC codes (occupations) to Work Styles ratings.
Download:	Work Styles.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Data Value	Float(5,2)	Rating associated with the O*NET-SOC occupation
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains the Work Styles Impact ratings and Distinctiveness Rank assignments for each O*NET-SOC occupation. Work Styles ratings are presented as two scales. WI reports the Impact rating of each Work Style on performance of an occupation's work activities and in relevant work contexts, from -3.00 (very detrimental) to +3.00 (very beneficial). DR reports the "distinctiveness rank" of a Work Style for an occupation, which presents up to 10 beneficial Work Styles which distinguish an occupation from others. A DR rating of 0.00 indicates the Work Style is not part of the ranked list. The file is displayed in seven tab delimited fields with the columns named O*NET-SOC Code, Element ID, Element Name, Scale ID, Data Value, Date, and Domain Source. The seven fields are represented by one row. There are a total of 37,422 rows of data in this file.

For more information, see:

Updating Higher-order Work Style Dimensions in the O*NET Work Styles Taxonomy
Revisiting the Work Styles Domain of the O*NET Content Model
Using a Hybrid Artificial Intelligence-Expert Method to Develop Work Style Ratings for the O*NET Database

File Structure Changes:
Release Number	Description of Change
5.0	Initial file addition
5.1	Added N, Standard Error, CI Bounds, Suppress columns
6.0 - 28.1	No structure changes
28.2	Standard Error, Lower CI Bound, Upper CI Bound expanded from 2 decimal places to 4
28.3 - 30.0	No structure changes
30.1	Removed N, Standard Error, CI Bounds, Suppress columns

Data Example - Work Styles:
O*NET-SOC Code	Element ID	Element Name	Scale ID	Data Value	Date	Domain Source
29-1141.01	1.C.1.a	Innovation	WI	1.10	12/2025	AI/Expert
29-1141.01	1.C.1.b	Achievement Orientation	WI	1.98	12/2025	AI/Expert

### Work Values

Purpose:	Provide a mapping of O*NET-SOC codes (occupations) to Work Values ratings.
Download:	Work Values.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Data Value	Float(5,2)	Rating associated with the O*NET-SOC occupation
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains the Content Model Work Values data associated with each O*NET-SOC occupation. The column named Data Value provides both the mean extent rating (indicated by the value EX in the Scale ID column) and the top three high-point values for respondents endorsing each occupation (indicated by VH in the Scale ID Column). The high-point values represent the following elements:

High-point value mapping:
0.00 = No high point available
1.00 = Achievement
2.00 = Working Conditions
3.00 = Recognition
4.00 = Relationships
5.00 = Support
6.00 = Independence

The file is displayed in seven tab delimited fields with the columns named O*NET-SOC Code, Element ID, Element Name, Scale ID, Data Value, Date, and Domain Source. The seven fields are represented by one row. There are a total of 7,866 rows of data in this file.

For more information, see:

Second Generation Occupational Value Profiles for the O*NET System: Summary
Occupational Value Profiles for New and Emerging Occupations in the O*NET System: Summary

File Structure Changes:
Release Number	Description of Change
5.0	Date and Source columns added
5.1 - 30.1	No structure changes

Data Example - Work Values:
O*NET-SOC Code	Element ID	Element Name	Scale ID	Data Value	Date	Domain Source
19-3033.00	1.B.2.a	Achievement	EX	5.83	11/2020	Analyst - Transition
19-3033.00	1.B.2.b	Working Conditions	EX	5.75	11/2020	Analyst - Transition
19-3033.00	1.B.2.c	Recognition	EX	5.33	11/2020	Analyst - Transition
19-3033.00	1.B.2.d	Relationships	EX	6.83	11/2020	Analyst - Transition
19-3033.00	1.B.2.e	Support	EX	3.17	11/2020	Analyst - Transition
19-3033.00	1.B.2.f	Independence	EX	6.00	11/2020	Analyst - Transition
19-3033.00	1.B.2.g	First Work Value High-Point	VH	4.00	11/2020	Analyst - Transition
19-3033.00	1.B.2.h	Second Work Value High-Point	VH	6.00	11/2020	Analyst - Transition
19-3033.00	1.B.2.i	Third Work Value High-Point	VH	1.00	11/2020	Analyst - Transition

## Tasks

### Task Statements

Purpose:	Provide a mapping of O*NET-SOC codes (occupations) to tasks associated with the occupation.
Download:	Task Statements.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Task ID	Integer(8)	Unique identifier for each task
Task	Character Varying(1000)	Task statement description
Task Type	Character Varying(12)	"Core" or "Supplemental" designation
Incumbents Responding	Integer(4)	Count of workers providing data
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains the tasks associated with each O*NET-SOC occupation. The 'Task Type' column identifies two categories of tasks: Core — tasks that are critical to the occupation. The criteria for these tasks are (a) relevance ≥ 67% and (b) a mean importance rating of ≥ 3.0. Supplemental — tasks that are less relevant and/or important to the occupation. Two sets of tasks are included in this category: (a) tasks rated ≥ 67% on relevance but < 3.0 on importance, and (b) tasks rated < 67% on relevance, regardless of mean importance. The file is displayed in seven tab delimited fields with the columns named O*NET-SOC Code, Task ID, Task, Task Type, Incumbents Responding, Date, and Domain Source. The seven fields are represented by one row. There are a total of 18,796 rows of data in this file.

For more information, see:

Summary of Procedures for O*NET Task Updating and New Task Generation

File Structure Changes:
Release Number	Description of Change
13.0	Added as a new file
14.0 - 30.1	No structure changes

Data Example - Task Statements:
O*NET-SOC Code	Task ID	Task	Task Type	Incumbents Responding	Date	Domain Source
29-1212.00	22689	Administer emergency cardiac care for life-threatening heart problems, such as cardiac arrest and heart attack.	n/a	n/a	12/2025	Analyst
29-1212.00	22690	Advise patients and community members concerning diet, activity, hygiene, or disease prevention.	n/a	n/a	12/2025	Analyst
29-1212.00	22691	Answer questions that patients have about their health and well-being.	n/a	n/a	12/2025	Analyst
29-1212.00	22692	Calculate valve areas from blood flow velocity measurements.	n/a	n/a	12/2025	Analyst
29-1212.00	22693	Compare measurements of heart wall thickness and chamber sizes to standards to identify abnormalities, using the results of an echocardiogram.	n/a	n/a	12/2025	Analyst
29-1212.00	22694	Conduct electrocardiogram (EKG), phonocardiogram, echocardiogram, or other cardiovascular tests to record patients' cardiac activity, using specialized electronic test equipment, recording devices, or laboratory instruments.	n/a	n/a	12/2025	Analyst
29-1212.00	22695	Conduct exercise electrocardiogram tests to monitor cardiovascular activity under stress.	n/a	n/a	12/2025	Analyst

### Task Ratings

Purpose:	Provide a mapping of O*NET-SOC codes (occupations) to the ratings for tasks associated with each occupation.
Download:	Task Ratings.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Task ID	Integer(8)	Unique task identifier (see Task Statements)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Category	Integer(3)	Percent frequency category (see Task Categories)
Data Value	Float(5,2)	Rating associated with the O*NET-SOC occupation
N	Integer(4)	Sample size
Standard Error	Float(7,4)	Standard Error
Lower CI Bound	Float(7,4)	Lower 95% confidence interval bound
Upper CI Bound	Float(7,4)	Upper 95% confidence interval bound
Recommend Suppress	Character(1)	Low precision indicator (Y=yes, N=no)
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains the task ratings associated with each O*NET-SOC occupation. It is displayed in 12 tab delimited fields and identified using the column names provided above. Item rating level metadata is provided in columns named N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, Date, and Domain Source. Refer to Appendix 2, Item Rating Level Statistics - Incumbent for additional information on these items. The 12 fields are represented by one row. There are a total of 161,559 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
13.0	Added as a new file
14.0 - 28.1	No structure changes
28.2	Standard Error, Lower CI Bound, Upper CI Bound expanded from 2 decimal places to 4
28.3 - 30.1	No structure changes

Data Example - Task Ratings:
O*NET-SOC Code	Task ID	Scale ID	Category	Data Value	N	Standard Error	Lower CI Bound	Upper CI Bound	Recommend Suppress	Date	Domain Source
53-3053.00	23756	FT	1	0.00	61	0.0000	n/a	n/a	N	08/2025	Incumbent
53-3053.00	23756	FT	2	0.24	61	0.1717	0.0546	1.0082	N	08/2025	Incumbent
53-3053.00	23756	FT	3	4.25	61	2.8703	1.0727	15.3926	N	08/2025	Incumbent
53-3053.00	23756	FT	4	4.89	61	3.7881	0.9999	20.7683	N	08/2025	Incumbent
53-3053.00	23756	FT	5	83.26	61	11.2185	49.8527	96.1377	N	08/2025	Incumbent
53-3053.00	23756	FT	6	0.34	61	0.2578	0.0721	1.5488	N	08/2025	Incumbent
53-3053.00	23756	FT	7	7.02	61	7.1452	0.8384	40.2745	N	08/2025	Incumbent
53-3053.00	23756	IM	n/a	4.84	62	0.0722	4.6956	4.9844	N	08/2025	Incumbent

### Task Categories

Purpose:	Provide description of Task categories.
Download:	Task Categories.txt
Structure and Description:
Column	Type	Column Content
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Category	Integer(3)	Category value associated with Scale ID
Category Description	Character Varying(1000)	Detail description of category associated with Scale ID

This file contains the categories associated with the Task content area. Categories for the scale Frequency of Task (FT) are included. The file is displayed in three tab delimited fields with the columns named Scale ID, Category, and Category Description. The three fields are represented by one row. There are a total of 7 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
13.0	Added as a new file
14.0 - 30.1	No structure changes

Data Example - Task Categories:
Scale ID	Category	Category Description
FT	1	Yearly or less
FT	2	More than yearly
FT	3	More than monthly
FT	4	More than weekly
FT	5	Daily
FT	6	Several times daily
FT	7	Hourly or more

### Emerging Tasks

Purpose:	Provide emerging task data associated with some O*NET-SOC occupations.
Download:	Emerging Tasks.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Task	Character Varying(1000)	New or revised task for an occupation
Category	Character Varying(8)	"New" or "Revision" designation
Original Task ID	Integer(8)	ID referencing the original task statement
Original Task	Character Varying(1000)	Previous task being modified
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains new and revised task statements proposed for future data collection. Statements are developed by analysts from sources including feedback from surveyed job incumbents, research into emerging technologies, and information provided by professional associations. The file is displayed in seven tab delimited fields with the columns named O*NET-SOC Code, Task, Category, Original Task ID, Original Task, Date, and Domain Source. The seven fields are represented by one row. There are a total of 328 rows of data in this file.

For more information, see:

Identification of Emerging Tasks in the O*NET System: A Revised Approach
Adding Drone-Specific Tasks to the O*NET Database: Initial Identification of Emerging Tasks using ChatGPT

File Structure Changes:
Release Number	Description of Change
20.1	Added as a new file
20.2 - 28.3	No structure changes
29.0	Removed Write-in Total column
29.1 - 30.1	No structure changes

Data Example - Emerging Tasks:
O*NET-SOC Code	Task	Category	Date	Domain Source
39-9031.00	Adjust workout programs and provide variations to address injuries or muscle soreness	New	08/2025	Occupational Expert
29-2011.00	Conduct blood typing and antibody screening	New	08/2025	Incumbent

## Technology Skills

### Technology Skills

Purpose:	Provide Technology Skills examples.
Download:	Technology Skills.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Example	Character Varying(150)	Specific technology skill illustration
Commodity Code	Integer(8)	UNSPSC commodity classification code (see UNSPSC Reference)
Commodity Title	Character Varying(150)	UNSPSC commodity category label
Hot Technology	Character(1)	Y/N indicator for widespread employer demand
In Demand	Character(1)	Y/N indicator for occupation-specific requirements

This file contains the Technology Skills examples, including hot and in-demand technologies, associated with O*NET-SOC occupations. The columns 'Commodity Code' and 'Commodity Title' classify the example under the United Nations Standard Products and Services Code (UNSPSC). See the UNSPSC Reference section for more information. The 'Hot Technology' column indicates requirements frequently included across all employer job postings. A concise list of all hot technologies may be downloaded from O*NET OnLine. The 'In Demand' column indicates requirements frequently included in employer job postings for the particular occupation. The file is displayed in six tab delimited fields with the columns named O*NET-SOC Code, Example, Commodity Code, Commodity Title, Hot Technology, and In Demand. The six fields are represented by one row. There are a total of 32,773 rows of data in this file.

For more information, see:

Hot Technologies and In Demand Technology Skills within the O*NET System
O*NET Center Tools and Technology Quality Control Processes
O*NET Tools and Technology: A Synopsis of Data Development Procedures
Identification of "Hot Technologies" within the O*NET System
Tools and Technology Search

File Structure Changes:
Release Number	Description of Change
23.2	Added as a new file
23.3 - 27.0	No structure changes
27.1	Added "In Demand" column
27.2 - 30.1	No structure changes

Data Example - Technology Skills:
O*NET-SOC Code	Example	Commodity Code	Commodity Title	Hot Technology	In Demand
11-2011.00	Actuate BIRT	43232314	Business intelligence and data analysis software	N	N
11-2011.00	Adobe Acrobat	43232202	Document management software	Y	N
11-2011.00	Adobe Acrobat Reader	43232202	Document management software	N	N
11-2011.00	Adobe After Effects	43232103	Video creation and editing software	Y	N
11-2011.00	Adobe Creative Cloud software	43232102	Graphics or photo imaging software	Y	N

### UNSPSC Reference

Purpose:	Provide relevant aspects of the UNSPSC taxonomy.
Download:	UNSPSC Reference.txt
Structure and Description:
Column	Type	Column Content
Commodity Code	Integer(8)	UNSPSC commodity code
Commodity Title	Character Varying(150)	UNSPSC commodity title
Class Code	Integer(8)	UNSPSC class code
Class Title	Character Varying(150)	UNSPSC class title
Family Code	Integer(8)	UNSPSC family code
Family Title	Character Varying(150)	UNSPSC family title
Segment Code	Integer(8)	UNSPSC segment code
Segment Title	Character Varying(150)	UNSPSC segment title

This file contains a listing of commodities in the United Nations Standard Products and Services Code (UNSPSC), version 260801. The UNSPSC is a four-level taxonomy for the classification of products and services, provided by the United Nations Development Programme. In the taxonomy, the Segment is the most general element and the Commodity is the most specific.

Hierarchy Example:
Segment: 43000000 - Information Technology Broadcasting and Telecommunications
Family: 43230000 - Software
Class: 43232100 - Content authoring and editing software
Commodity: 43232104 - Word processing software

The file is displayed in 8 tab delimited fields with the columns named Commodity Code, Commodity Title, Class Code, Class Title, Family Code, Family Title, Segment Code, and Segment Title. The 8 fields are represented by one row. There are a total of 4,264 rows of data in this file.

For more information, see:

O*NET Center Tools and Technology Quality Control Processes
O*NET Tools and Technology: A Synopsis of Data Development Procedures
Identification of "Hot Technologies" within the O*NET System

File Structure Changes:
Release Number	Description of Change
20.1	Added as a new file
20.2 - 30.1	No structure changes

Data Example - UNSPSC Reference:
Commodity Code	Commodity Title	Class Code	Class Title	Family Code	Family Title	Segment Code	Segment Title
12131704	Explosive initiators	12131700	Igniters	12130000	Explosive materials	12000000	Chemicals including Bio Chemicals and Gas Materials
12131707	Lighters	12131700	Igniters	12130000	Explosive materials	12000000	Chemicals including Bio Chemicals and Gas Materials
14111513	Ledger paper	14111500	Printing and writing paper	14110000	Paper products	14000000	Paper Materials and Products
14111802	Receipts or receipt books	14111800	Business use papers	14110000	Paper products	14000000	Paper Materials and Products

### Tools Used

Purpose:	Provide Tools Used examples.
Download:	Tools Used.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Example	Character Varying(150)	Specific tool or equipment instance
Commodity Code	Integer(8)	UNSPSC categorization number (see UNSPSC Reference)
Commodity Title	Character Varying(150)	UNSPSC classification label

This file contains the Tools Used examples associated with O*NET-SOC occupations. The columns 'Commodity Code' and 'Commodity Title' classify the example under the United Nations Standard Products and Services Code (UNSPSC). See the UNSPSC Reference section for more information. The file is displayed in four tab delimited fields with the columns named O*NET-SOC Code, Example, Commodity Code, and Commodity Title. The four fields are represented by one row. There are a total of 41,662 rows of data in this file.

For more information, see:

O*NET Center Tools and Technology Quality Control Processes
O*NET Tools and Technology: A Synopsis of Data Development Procedures
Tools and Technology Search

File Structure Changes:
Release Number	Description of Change
23.2	Added as a new file
23.3 - 30.1	No structure changes

Data Example - Tools Used:
O*NET-SOC Code	Example	Commodity Code	Commodity Title
11-2011.00	Computer data input scanners	43211711	Scanners
11-2011.00	Desktop computers	43211507	Desktop computers
11-2011.00	Laptop computers	43211503	Notebook computers

## Work Activities

### Work Activities

Purpose:	Provide a mapping of O*NET-SOC codes (occupations) to Work Activity ratings.
Download:	Work Activities.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Data Value	Float(5,2)	Rating associated with the O*NET-SOC occupation
N	Integer(4)	Sample size
Standard Error	Float(7,4)	Standard Error
Lower CI Bound	Float(7,4)	Lower 95% confidence interval bound
Upper CI Bound	Float(7,4)	Upper 95% confidence interval bound
Recommend Suppress	Character(1)	Low precision indicator (Y=yes, N=no)
Not Relevant	Character(1)	Not relevant for the occupation (Y=yes, N=no)
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains the Content Model Work Activity data associated with each O*NET-SOC occupation. It is displayed in 13 tab delimited fields and identified using the column names provided above. Item rating level metadata is provided in columns named N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, Not Relevant, Date, and Domain Source. Refer to Appendix 2, Item Rating Level Statistics - Incumbent for additional information on these items. The 13 fields are represented by one row. There are a total of 73,308 rows of data in this file.

For more information, see:

O*NET Work Activities Project Technical Report

File Structure Changes:
Release Number	Description of Change
5.0	Date and Source columns added
5.1	Columns added for N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, and Not Relevant
6.0 - 28.1	No structure changes
28.2	Standard Error, Lower CI Bound, Upper CI Bound expanded from 2 decimal places to 4
28.3 - 30.1	No structure changes

Data Example - Work Activities:
O*NET-SOC Code	Element ID	Element Name	Scale ID	Data Value	N	Standard Error	Lower CI Bound	Upper CI Bound	Recommend Suppress	Not Relevant	Date	Domain Source
17-2121.00	4.A.1.a.1	Getting Information	IM	4.22	18	n/a	n/a	n/a	n/a	n/a	08/2025	Occupational Expert
17-2121.00	4.A.1.a.1	Getting Information	LV	5.17	18	n/a	n/a	n/a	n/a	N	08/2025	Occupational Expert
17-2121.00	4.A.1.a.2	Monitoring Processes, Materials, or Surroundings	IM	3.12	17	n/a	n/a	n/a	n/a	n/a	08/2025	Occupational Expert
17-2121.00	4.A.1.a.2	Monitoring Processes, Materials, or Surroundings	LV	3.94	17	n/a	n/a	n/a	n/a	N	08/2025	Occupational Expert
17-2121.00	4.A.1.b.1	Identifying Objects, Actions, and Events	IM	3.83	18	n/a	n/a	n/a	n/a	n/a	08/2025	Occupational Expert
17-2121.00	4.A.1.b.1	Identifying Objects, Actions, and Events	LV	5.00	18	n/a	n/a	n/a	n/a	N	08/2025	Occupational Expert
17-2121.00	4.A.1.b.2	Inspecting Equipment, Structures, or Materials	IM	3.76	17	n/a	n/a	n/a	n/a	n/a	08/2025	Occupational Expert
17-2121.00	4.A.1.b.2	Inspecting Equipment, Structures, or Materials	LV	4.35	17	n/a	n/a	n/a	n/a	N	08/2025	Occupational Expert

### IWA Reference

Purpose:	Provide each Intermediate Work Activity.
Download:	IWA Reference.txt
Structure and Description:
Column	Type	Column Content
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
IWA ID	Character Varying(20)	Identifies each Intermediate Work Activity
IWA Title	Character Varying(150)	Intermediate Work Activity statement

This file contains each Intermediate Work Activity and its corresponding O*NET Work Activity element ID. Every IWA is linked to exactly one Work Activity from the O*NET Content Model. IWAs are linked to one or more DWAs; see the DWA Reference file for these links. The file is displayed in three tab delimited fields with the columns named Element ID, IWA ID, and IWA Title. The three fields are represented by one row. There are a total of 332 rows of data in this file.

For more information, see:

O*NET Work Activities Project Technical Report

File Structure Changes:
Release Number	Description of Change
18.1	Added as a new file
19.0 - 30.1	No structure changes

Data Example - IWA Reference:
Element ID	IWA ID	IWA Title
4.A.1.a.1	4.A.1.a.1.I01	Study details of artistic productions
4.A.1.a.1	4.A.1.a.1.I02	Read documents or materials to inform work processes
4.A.2.b.2	4.A.2.b.2.I14	Design industrial systems or equipment
4.A.4.c.2	4.A.4.c.2.I01	Perform recruiting or hiring activities

### DWA Reference

Purpose:	Provide each Detailed Work Activity.
Download:	DWA Reference.txt
Structure and Description:
Column	Type	Column Content
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
IWA ID	Character Varying(20)	Intermediate Work Activity identifier (see IWA Reference)
DWA ID	Character Varying(20)	Detailed Work Activity identifier
DWA Title	Character Varying(150)	Detailed Work Activity statement

This file contains each Detailed Work Activity and its corresponding GWA and IWA identifiers. Each DWA is linked to exactly one IWA, which in turn is linked to exactly one Work Activity from the O*NET Content Model. See Content Model Reference and IWA Reference for information about these higher-level elements. Each DWA is linked to multiple task statements; see Tasks to DWAs for these links. The file is displayed in four tab delimited fields with the columns named Element ID, IWA ID, DWA ID, and DWA Title. The four fields are represented by one row. There are a total of 2,087 rows of data in this file.

For more information, see:

O*NET Work Activities Project Technical Report
Ranking Detailed Work Activities (DWAs) Within O*NET Occupational Profiles

File Structure Changes:
Release Number	Description of Change
18.1	Added as a new file
19.0 - 30.1	No structure changes

Data Example - DWA Reference:
Element ID	IWA ID	DWA ID	DWA Title
4.A.1.a.1	4.A.1.a.1.I01	4.A.1.a.1.I01.D01	Review art or design materials
4.A.1.a.1	4.A.1.a.1.I01	4.A.1.a.1.I01.D02	Study details of musical compositions
4.A.2.b.2	4.A.2.b.2.I14	4.A.2.b.2.I14.D06	Design control systems for mechanical equipment
4.A.4.b.6	4.A.4.b.6.I09	4.A.4.b.6.I09.D03	Advise others on health and safety issues

### Tasks to DWAs

Purpose:	Provide a mapping of task statements to Detailed Work Activities.
Download:	Tasks to DWAs.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Task ID	Integer(8)	Identifies each task (see Task Statements)
DWA ID	Character Varying(20)	Identifies each Detailed Work Activity (see DWA Reference)
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file maps each Detailed Work Activity (DWA) to the task statements, and consequently to the O*NET-SOC occupations, requiring that activity. Each DWA is mapped to multiple task statements, and each referenced task statement is mapped to one or more DWAs. The file is displayed in five tab delimited fields with the columns named O*NET-SOC Code, Task ID, DWA ID, Date, and Domain Source. The five fields are represented by one row. There are a total of 23,850 rows of data in this file.

For more information, see:

O*NET Work Activities Project Technical Report

File Structure Changes:
Release Number	Description of Change
18.1	Added as a new file
19.0 - 30.1	No structure changes

Data Example - Tasks to DWAs:
O*NET-SOC Code	Task ID	DWA ID	Date	Domain Source
25-3011.00	6824	4.A.3.b.6.I12.D04	03/2014	Analyst
25-3011.00	6825	4.A.1.a.2.I06.D03	03/2014	Analyst
25-3011.00	6825	4.A.2.a.1.I03.D04	03/2014	Analyst
25-3011.00	6826	4.A.2.b.2.I15.D06	03/2014	Analyst
25-3011.00	6827	4.A.4.b.3.I02.D06	03/2014	Analyst

## Work Context

### Work Context

Purpose:	Provide a mapping of O*NET-SOC codes (occupations) to Work Context ratings.
Download:	Work Context.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Category	Integer(3)	Percent frequency category (see Work Context Categories)
Data Value	Float(5,2)	Rating associated with the O*NET-SOC occupation
N	Integer(4)	Sample size
Standard Error	Float(7,4)	Standard Error
Lower CI Bound	Float(7,4)	Lower 95% confidence interval bound
Upper CI Bound	Float(7,4)	Upper 95% confidence interval bound
Recommend Suppress	Character(1)	Low precision indicator (Y=yes, N=no)
Not Relevant	Character(1)	Not relevant for the occupation (Y=yes, N=no)
Date	Character(7)	Date when data was updated
Domain Source	Character Varying(30)	Source of the data

This file contains the Content Model Work Context data associated with each O*NET-SOC occupation. It is displayed in 14 tab delimited fields and identified using the column names provided above. Item rating level metadata is provided in columns named N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, Not Relevant, Date, and Domain Source. Refer to Appendix 2, Item Rating Level Statistics - Incumbent for additional information on these items. The 14 fields are represented by one row. There are a total of 297,676 rows of data in this file. The column named Data Value provides both the mean rating (indicated by the value CX in the Scale ID column) and the percent of respondents endorsing each category (indicated by CXP in the Scale ID column).

File Structure Changes:
Release Number	Description of Change
5.0	Date and Source columns added
5.1	Columns added for N, Standard Error, Lower CI Bound, Upper CI Bound, Recommend Suppress, and Not Relevant
6.0 - 28.1	No structure changes
28.2	Standard Error, Lower CI Bound, Upper CI Bound expanded from 2 decimal places to 4
28.3 - 30.1	No structure changes

Data Example - Work Context:
O*NET-SOC Code	Element ID	Element Name	Scale ID	Category	Data Value	N	Standard Error	Lower CI Bound	Upper CI Bound	Recommend Suppress	Not Relevant	Date	Domain Source
47-2141.00	4.C.3.d.8	Duration of Typical Work Week	CT	n/a	1.99	20	0.2281	1.5163	2.4712	N	n/a	08/2025	Incumbent
47-2141.00	4.C.3.d.8	Duration of Typical Work Week	CTP	1	17.03	20	17.7643	1.4564	74.0353	Y	n/a	08/2025	Incumbent
47-2141.00	4.C.3.d.8	Duration of Typical Work Week	CTP	2	66.56	20	23.7009	17.6484	94.8695	Y	n/a	08/2025	Incumbent
47-2141.00	4.C.3.d.8	Duration of Typical Work Week	CTP	3	16.41	20	15.0155	1.9455	65.9997	Y	n/a	08/2025	Incumbent

### Work Context Categories

Purpose:	Provide descriptions of Work Context categories.
Download:	Work Context Categories.txt
Structure and Description:
Column	Type	Column Content
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Category	Integer(3)	Category value associated with element
Category Description	Character Varying(1000)	Detail description of category associated with element

This file contains the categories associated with the Work Context content area. Categories for the following scales are included: Context (CXP) and Context Category (CTP). The file includes categories utilized in the data collection survey where the category descriptions are variable and item specific. It is displayed in 5 tab delimited fields. There are a total of 281 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
9.0	Added as a new file
10.0 - 30.1	No structure changes

Data Example - Work Context Categories:
The file includes entries such as "Face-to-Face Discussions" with five frequency categories (Never through Every day) and "Contact With Others" with five intensity levels (No contact through Constant contact).

## Occupation Titles

### Occupation Data

Purpose:	Provide O*NET-SOC codes, titles, and descriptions.
Download:	Occupation Data.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code
Title	Character Varying(150)	O*NET-SOC Title
Description	Character Varying(1000)	O*NET-SOC Description

This file contains each O*NET-SOC code, occupational title, and definition/description. The file is displayed in three tab delimited fields with the columns named O*NET-SOC Code, Title, and Description. The three fields are represented by one row. There are a total of 1,016 rows of data in this file.

For more information, see:

Updating the O*NET-SOC Taxonomy: Incorporating the 2010 SOC Structure

File Structure Changes:
Release Number	Description of Change
5.0 - 30.1	No structure changes

Data Example - Occupation Data:
O*NET-SOC Code	Title	Description
11-9041.01	Biofuels/Biodiesel Technology and Product Development Managers	Define, plan, or execute biofuels/biodiesel research programs...
17-2072.00	Electronics Engineers, Except Computer	Research, design, develop, or test electronic components...
19-4031.00	Chemical Technicians	Conduct chemical and physical laboratory tests...
45-4011.00	Forest and Conservation Workers	Under supervision, perform manual labor necessary to...
51-8012.00	Power Distributors and Dispatchers	Coordinate, regulate, or distribute electricity...

### Alternate Titles

Purpose:	Provide alternate occupational titles for O*NET-SOC occupations.
Download:	Alternate Titles.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Alternate Title	Character Varying(250)	Alternative occupational designation
Short Title	Character Varying(150)	Abbreviated version (when applicable)
Source(s)	Character Varying(50)	Source code reference list

This file contains job or alternate 'lay' titles linked to occupations in the O*NET-SOC classification system. The file was developed to improve keyword searches in several Department of Labor internet applications (i.e., Career InfoNet, O*NET OnLine, and O*NET Code Connector). The file contains occupational titles from existing occupational classification systems, as well as from other diverse sources. When a title contains acronyms, abbreviations, or jargon, the 'Short Title' column contains the brief version of the full title. The 'Source(s)' column contains a comma delimited list of codes which indicate the source of the title information; the codes are identified below:

Source Codes Reference:
01 = Industry associations
02 = Incumbent worker surveys
03 = Occupational classification assignments
04 = Standard SOC resources
05 = State agencies
06 = Census Bureau materials
07 = BLS databases
08 = ETA sources
09 = User submissions
10 = Employer job listings

The file is displayed in four tab delimited fields with the columns named O*NET-SOC Code, Alternate Title, Short Title, and Source(s). The four fields are represented by one row. There are a total of 56,505 rows of data in this file.

For more information, see:

O*NET Alternate Titles Procedures
A Weighted O*NET Keyword Search (WWS)
Military Transition Search (as used in My Next Move for Veterans)

File Structure Changes:
Release Number	Description of Change
20.1	Added as a new file
20.2 - 21.3	No structure changes
22.0	Alternate Title field expanded from 150 to 250 characters
22.1 - 30.1	No structure changes

Data Example - Alternate Titles:
O*NET-SOC Code	Alternate Title	Short Title	Source(s)
29-2099.00	Sleep Technician	n/a	09

### Sample of Reported Titles

Purpose:	Provide job titles reported during O*NET data collection.
Download:	Sample of Reported Titles.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Reported Job Title	Character Varying(150)	Titles submitted by workers or subject matter experts
Shown in My Next Move	Character(1)	Display indicator (Y=yes, N=no)

This file contains job titles frequently reported by incumbents and occupational experts on data collection surveys. These titles are displayed on occupational reports in the O*NET OnLine and O*NET Code Connector web applications; up to 10 titles for each occupation are displayed and included in this file. Up to 4 titles are also displayed in My Next Move, My Next Move for Veterans, and Mi Próximo Paso; the titles shown in these applications are marked with a Y in the "Shown in My Next Move" column. The file is displayed in three tab delimited fields with the columns named O*NET-SOC Code, Reported Job Title, and Shown in My Next Move. The three fields are represented by one row. There are a total of 7,955 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
20.1	Added as a new file
20.2 - 30.1	No structure changes

Data Example - Sample of Reported Titles:
Sample entries for O*NET-SOC 17-2071.00 (Electrical Engineer) include titles such as Design Engineer, Electrical Engineer, and Electrical Design Engineer, with selective inclusion in My Next Move indicated.

## Related Occupations and Related Domains

### Related Occupations

Purpose:	Provide related occupation links between O*NET-SOC occupations.
Download:	Related Occupations.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Related O*NET-SOC Code	Character(10)	Associated O*NET-SOC code mapping
Relatedness Tier	Character Varying(50)	Categories indicating similarity level
Index	Integer(3)	Ranking based on expert review

For each O*NET-SOC code included, 10 primary and 10 supplemental related O*NET-SOC codes are listed. The related occupations in this file are developed using an approach which includes three important contributors to occupational similarity: what people in the occupations do, what they know, and what they are called.

Relatedness Tier Categories:
Primary-Short: Five most strongly related occupations after expert review
Primary-Long: 6th to 10th most strongly related occupations after expert review
Supplemental: 11th to 20th most strongly related occupations after expert review

The file is displayed in four tab delimited fields with the columns named O*NET-SOC Code, Related O*NET-SOC Code, Relatedness Tier, and Index. The four fields are represented by one row. There are a total of 18,460 rows of data in this file.

For more information, see:

Developing Related Occupations for the O*NET Program
Updates to Related Occupations for the O*NET Program Using the O*NET 28.0 Database

File Structure Changes:
Release Number	Description of Change
26.3	Added as a new file
27.0 - 30.1	No structure changes

Data Example - Related Occupations:
O*NET-SOC Code	Related O*NET-SOC Code	Relatedness Tier	Index
17-1011.00	17-1012.00	Primary-Short	1
17-1011.00	11-9021.00	Primary-Short	2
17-1011.00	27-1025.00	Primary-Short	3

### Abilities to Work Activities

Purpose:	Provide linkages between abilities and relevant work activities.
Download:	Abilities to Work Activities.txt
Structure and Description:
Column	Type	Column Content
Abilities Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Abilities Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Work Activities Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Work Activities Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)

This file contains linkages between abilities and relevant work activities. Occupation-specific ratings for the listed elements may be found in the Abilities and Work Activities files. Linkages were developed by a panel of experienced industrial/organizational psychologists, and are used in the development of analyst occupational abilities ratings. It is displayed in 4 tab delimited fields. There are a total of 381 rows of data in this file.

For more information, see:

O*NET Analyst Occupational Abilities Ratings: Procedures Update

File Structure Changes:
Release Number	Description of Change
24.2	Added as a new file
24.3 - 30.1	No structure changes

Data Example - Abilities to Work Activities:
Abilities Element ID	Abilities Element Name	Work Activities Element ID	Work Activities Element Name
1.A.1.a.1	Oral Comprehension	4.A.1.a.1	Getting Information
1.A.1.a.1	Oral Comprehension	4.A.1.a.2	Monitoring Processes, Materials, or Surroundings
1.A.1.a.1	Oral Comprehension	4.A.1.b.1	Identifying Objects, Actions, and Events

### Abilities to Work Context

Purpose:	Provide linkages between abilities and relevant work context.
Download:	Abilities to Work Context.txt
Structure and Description:
Column	Type	Column Content
Abilities Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Abilities Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Work Context Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Work Context Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)

This file contains linkages between abilities and relevant work context. Occupation-specific ratings for the listed elements may be found in the Abilities and Work Context files. Linkages were developed by a panel of experienced industrial/organizational psychologists, and are used in the development of analyst occupational abilities ratings. It is displayed in 4 tab delimited fields. There are a total of 139 rows of data in this file.

For more information, see:

O*NET Analyst Occupational Abilities Ratings: Procedures Update

File Structure Changes:
Release Number	Description of Change
24.2	Added as a new file
24.3 - 30.1	No structure changes

Data Example - Abilities to Work Context:
Abilities Element ID	Abilities Element Name	Work Context Element ID	Work Context Element Name
1.A.1.a.1	Oral Comprehension	4.C.1.a.2.c	Public Speaking
1.A.1.a.1	Oral Comprehension	4.C.1.a.2.f	Telephone Conversations
1.A.1.a.1	Oral Comprehension	4.C.1.a.2.l	Face-to-Face Discussions

### Skills to Work Activities

Purpose:	Provide linkages between skills and relevant work activities.
Download:	Skills to Work Activities.txt
Structure and Description:
Column	Type	Column Content
Skills Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Skills Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Work Activities Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Work Activities Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)

This file contains linkages between skills and relevant work activities. Occupation-specific ratings for the listed elements may be found in the Skills and Work Activities files. Linkages were developed by a panel of experienced industrial/organizational psychologists, and are used in the development of analyst occupational skills ratings. It is displayed in 4 tab delimited fields. There are a total of 232 rows of data in this file.

For more information, see:

O*NET Analyst Occupational Skills Ratings: Procedures Update

File Structure Changes:
Release Number	Description of Change
24.2	Added as a new file
24.3 - 30.1	No structure changes

Data Example - Skills to Work Activities:
Skills Element ID	Skills Element Name	Work Activities Element ID	Work Activities Element Name
2.A.1.a	Reading Comprehension	4.A.1.a.1	Getting Information
2.A.1.a	Reading Comprehension	4.A.1.a.2	Monitoring Processes, Materials, or Surroundings
2.A.1.a	Reading Comprehension	4.A.1.b.1	Identifying Objects, Actions, and Events
2.A.1.a	Reading Comprehension	4.A.2.a.1	Judging the Qualities of Objects, Services, or People
2.A.1.a	Reading Comprehension	4.A.2.a.2	Processing Information

### Skills to Work Context

Purpose:	Provide linkages between skills and relevant work context.
Download:	Skills to Work Context.txt
Structure and Description:
Column	Type	Column Content
Skills Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Skills Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Work Context Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Work Context Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)

This file contains linkages between skills and relevant work context. Occupation-specific ratings for the listed elements may be found in the Skills and Work Context files. Linkages were developed by a panel of experienced industrial/organizational psychologists, and are used in the development of analyst occupational skills ratings. It is displayed in 4 tab delimited fields. There are a total of 96 rows of data in this file.

For more information, see:

O*NET Analyst Occupational Skills Ratings: Procedures Update

File Structure Changes:
Release Number	Description of Change
24.2	Added as a new file
24.3 - 30.1	No structure changes

Data Example - Skills to Work Context:
Skills Element ID	Skills Element Name	Work Context Element ID	Work Context Element Name
2.A.1.a	Reading Comprehension	4.C.1.a.2.h	E-Mail
2.A.1.b	Active Listening	4.C.1.a.2.c	Public Speaking
2.A.1.b	Active Listening	4.C.1.a.2.f	Telephone Conversations
2.A.1.b	Active Listening	4.C.1.a.2.l	Face-to-Face Discussions

## Data Collection

### Content Model Reference

Purpose:	Provide O*NET Content Model elements.
Download:	Content Model Reference.txt
Structure and Description:
Column	Type	Column Content
Element ID	Character Varying(20)	Content Model Outline Position
Element Name	Character Varying(150)	Content Model Element Name
Description	Character Varying(1500)	Content Model Element Description

This file contains the Content Model elements and descriptions. The file is displayed in three tab delimited fields with the columns named Element ID, Element Name, and Description. The three fields are represented by one row. There are a total of 630 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
5.0 - 30.1	No structure changes

Data Example - Content Model Reference:
Element ID	Element Name	Description
1	Worker Characteristics	Worker Characteristics
1.A	Abilities	Enduring attributes of the individual that influence performance
1.A.1	Cognitive Abilities	Abilities that influence the acquisition and application of knowledge in problem solving
1.A.1.a	Verbal Abilities	Abilities related to verbal information application in problem-solving contexts
1.A.1.a.1	Oral Comprehension	The ability to listen to and understand information and ideas presented through spoken words and sentences

### Occupation Level Metadata

Purpose:	Provide O*NET-SOC Occupational Level Metadata associated with the incumbent data collection.
Download:	Occupation Level Metadata.txt
Structure and Description:
Column	Type	Column Content
O*NET-SOC Code	Character(10)	O*NET-SOC Code (see Occupation Data)
Item	Character Varying(150)	Occupation-level statistical measures
Response	Character Varying(75)	Classification of response type
N	Integer(4)	Sample size for the occupation
Percent	Float(4,1)	Percentage distribution of responses
Date	Character(7)	Date when data was updated

This file contains occupational level metadata variables associated with data collection statistics. Refer to Appendix 3, Key to Occupation Level Metadata for additional descriptions of the data provided in this file. The file is displayed in six tab delimited fields with the columns named O*NET-SOC Code, Item, Response, N, Percent, and Date. The six fields are represented by one row. There are a total of 32,202 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
5.1	Added as a new file
6.0 - 20.3	No structure changes
21.0	Items added/renamed (see Appendix 3)
21.1 - 30.1	No structure changes

Data Example - Occupation Level Metadata:
O*NET-SOC Code	Item	Response	N	Percent	Date
17-2111.02	Data Collection Mode	Paper	26	15.4	08/2025
17-2111.02	Data Collection Mode	Web	26	84.6	08/2025
17-2111.02	How Much Experience Performing Work in this Occupation	1-2 Years	26	0.0	08/2025
17-2111.02	How Much Experience Performing Work in this Occupation	10+ Years	26	96.2	08/2025
17-2111.02	How Much Experience Performing Work in this Occupation	3-4 Years	26	0.0	08/2025
17-2111.02	How Much Experience Performing Work in this Occupation	5-9 Years	26	3.8	08/2025
17-2111.02	OE Completeness Rate	n/a	n/a	100.0	08/2025
17-2112.00	Data Collection Mode	Paper	84	42.9	08/2020

### Level Scale Anchors

Purpose:	Provide descriptions of O*NET Level Scale Anchors.
Download:	Level Scale Anchors.txt
Structure and Description:
Column	Type	Column Content
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)
Anchor Value	Integer(3)	Numeric value for anchor
Anchor Description	Character Varying(1000)	Detailed anchor description

This file contains the scale anchors associated with the following four content areas – 1) Abilities, 2) Knowledge, 3) Skills, and 4) Work Activities. It includes all scale anchors utilized in the data collection survey where the scale anchors are variable and item specific. Scale anchors are not included for those survey items where the scale anchors are fixed. This includes the five-point importance scale and the seven-point task frequency scale. The file is displayed in five tab delimited fields with the columns named Element ID, Element Name, Scale ID, Anchor Value, and Anchor Description. The five fields are represented by one row. There are a total of 483 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
5.1	Added as a new file
6.0	Added Scale ID column
7.0 - 8.0	No structure changes
9.0	Education, Training, Experience, and Work Context data relocated
10.0 - 30.1	No structure changes

Data Example - Level Scale Anchors:
Sample entries demonstrating progression for Oral Comprehension:
Anchor Value 2: Understand a television commercial
Anchor Value 4: Understand a coach's oral instructions
Anchor Value 6: Understand an advanced physics lecture

### Scales Reference

Purpose:	Provide a reference to the scale names and values.
Download:	Scales Reference.txt
Structure and Description:
Column	Type	Column Content
Scale ID	Character Varying(3)	Scale ID
Scale Name	Character Varying(50)	Scale Name
Minimum	Integer(1)	Scale Minimum
Maximum	Integer(3)	Scale Maximum

This file contains the Scale information by which the raw values are measured. The file is displayed in four tab delimited fields with the columns named Scale ID, Scale Name, Minimum, and Maximum. The four fields are represented by one row. There are a total of 31 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
5.0 - 30.1	No structure changes

Data Example - Scales Reference:
Scale ID	Scale Name	Minimum	Maximum
CT	Context	1	3
CTP	Context (Categories 1-3)	0	100
CX	Context	1	5
CXP	Context (Categories 1-5)	0	100
IM	Importance	1	5
LV	Level	0	7
OJ	On-The-Job Training (Categories 1-9)	0	100
PT	On-Site Or In-Plant Training (Categories 1-9)	0	100
RL	Required Level Of Education (Categories 1-12)	0	100
RW	Related Work Experience (Categories 1-11)	0	100

### Survey Booklet Locations

Purpose:	Provide survey item numbers for O*NET Content Model elements.
Download:	Survey Booklet Locations.txt
Structure and Description:
Column	Type	Column Content
Element ID	Character Varying(20)	Content Model Outline Position (see Content Model Reference)
Element Name	Character Varying(150)	Content Model Element Name (see Content Model Reference)
Survey Item Number	Character Varying(5)	Survey Booklet Location Number
Scale ID	Character Varying(3)	Scale ID (see Scales Reference)

This file contains the Content Model elements that have corresponding survey item numbers in the Survey Booklet. Each survey item number corresponds to a survey question in the O*NET Questionnaires. The values for incumbent data categories are percentage ratings corresponding to survey question options. Match the element ID(s) from data files to a survey item number using this file. The file is displayed in four tab delimited fields with the columns named Element ID, Element Name, Survey Item Number, and Scale ID. The four fields are represented by one row. There are a total of 211 rows of data in this file.

File Structure Changes:
Release Number	Description of Change
5.0	Added as a new file
5.1 - 12.0	No structure changes
13.0	Added Scale ID column
14.0 - 29.1	No structure changes
29.2	Survey Item Number expanded from 4 to 5 characters
29.3 - 30.1	No structure changes

Data Example - Survey Booklet Locations:
Element ID	Element Name	Survey Item Number	Scale ID
2.C.1.a	Administration and Management	KN01	IM
2.C.1.a	Administration and Management	KN01b	LV
2.C.1.b	Administrative	KN02	IM
2.C.1.b	Administrative	KN02b	LV
2.C.1.c	Economics and Accounting	KN03	IM
2.C.1.c	Economics and Accounting	KN03b	LV

## Appendices

### Appendix 1. Item Rating Level Statistics - Analyst

This appendix documents statistical measures for Ability and Skills domain ratings collected from analyst respondents.

Key Statistical Measures:

Standard Error (SEM): The standard deviation of the ratings across analysts divided by the square root of the number of analysts (i.e., eight) indicates estimate precision.

Confidence Intervals: Upper and lower bounds calculated by multiplying SEM by 1.96 and adding/subtracting from the observed mean establish a 95% confidence range.

Suppress Recommendations: Estimates with standard errors exceeding .51 are flagged for caution, as the upper and lower bounds of the confidence interval are more than 1 scale point away from the observed mean.

Not Relevant Flag: Items receive a "Y" designation when fewer than 3 analysts rated importance at level 2 or above.

### Appendix 2. Item Rating Level Statistics - Incumbent

This appendix documents Item Rating Level Statistics for Incumbent data in O*NET 30.1.

Key Definitions:

Standard Error: Measures precision of estimates; calculated as the square root of variance. Smaller variances indicate greater precision.

Confidence Intervals: The 95% CI bounds create a range around estimates. For approximately 95 percent of the samples, the interval would include the "true" average value.

Recommend Suppress Flag: Applied when estimates have low precision, determined by: sample size <10, zero variance with sample <15, or relative standard error >0.5.

Not Relevant Flag: Applied when >75% of item respondents to the corresponding "Importance" item rated the item as "not important."

### Appendix 3. Key to Occupation Level Metadata

This appendix documents metadata fields associated with each O*NET-SOC code. The O*NET-SOC Level Sample Distribution Statistics are unweighted percents and therefore don't represent actual population distributions.

Data Collection Metadata Fields:

Data Collection Mode: Respondents could choose between "Paper" or "Web" survey formats.

Employee Completeness Rate: The percentage of total returned non-blank questionnaires that were retained after editing and data cleaning.

Employee Response Rate: The percentage of eligible employees in the sample who return a non-blank questionnaire.

Establishment Eligibility Rate: The percentage of sampled establishments where target occupations were found operational at the sampled address.

Establishment Response Rate: The percentage of sampled eligible establishments for the occupation that agreed to participate.

How Long at Current Job: Response options included tenure brackets (10+ years, 6-9 years, 1-5 years, <1 year) plus Missing.

How Much Experience Performing Work in this Occupation: Occupational experts selected from options ranging from "10+ years" through "<1 year," "Never," or "Missing."

Industry Division: Classifications using SIC codes across 11 categories from Agriculture through Public Administration.

NAICS Sector: Twenty economic activity classifications for sampled establishments.

OE Completeness/Response Rates: Metrics specific to occupational expert questionnaire retention and participation.

SOC Eligibility Rate: Percentage of eligible establishments where target occupations were present.

Total Completed Questionnaires: Count of incumbent questionnaire completions across all questionnaire types.

### Appendix 4. Content Updates Since Release 4.0

O*NET 30.1 Database (923 occupations updated):

Work Styles Updates: An updated Work Styles Taxonomy was implemented using a hybrid artificial intelligence-expert method to refresh data for 891 occupations.

Related Occupations: Related occupations for all 923 data-level O*NET-SOC occupations were identified through machine learning and analyst review.

Technology Skills: The database features an updated listing of 171 "Hot Technologies" sourced from employer job postings, creating over 11,500 occupation linkages. Additionally, 2,400+ "In Demand" skill connections were established across 495 occupations, with 120+ new technology linkages added.

Commodity Updates: Technology skill commodities were updated to reflect United Nations Standard Products and Services Code (UNSPSC-UNv260801), adding 2 new commodities and renaming 32 tool commodities affecting thousands of linkages.

Minor Updates: Alternate titles for 14 occupations were revised, and one occupation's task list underwent analyst review.

O*NET 30.0 Database (218 occupations): This release focused on comprehensive occupational data expansion, including abilities, skills, task ratings, work activities, knowledge, work context, training/experience, work styles, job zones, and detailed work activities across 78 occupations.

Earlier Releases (29.3 through 4.0): The database documents systematic updates spanning from Release 5.0 (first data collection release in 2003) through current versions, tracking the evolution from analyst-only ratings to incumbent-based data collection methodologies.

### Appendix 5. Historical Summary of Database Content Changes

This appendix documents changes to O*NET database content across multiple release cycles, tracking modifications to various occupational data files from version 5.0 through 30.1.

Change Type Legend:
U = Content updates from the data collection program
N = New data elements, types, or descriptors appearing for the first time
C = Other editorial revisions and corrections

Key Content Areas Tracked:

Competencies & Requirements: Knowledge, Skills, Abilities, Education/Training/Experience, Job Zones

Work Characteristics: Work Activities, Work Context, Work Styles, Task Statements, Task Ratings

Technology & Tools: Technology Skills, Tools Used, UNSPSC Reference

Career Information: Alternate Titles, Sample Reported Titles, Related Occupations

Reference Materials: Content Model Reference, Occupation Level Metadata, Scales Reference, Survey Booklet Locations

The tables present comprehensive tracking from release 5.0 through 30.1, allowing users to trace which files received updates in specific versions. Files with shaded cells indicate they did not exist during that particular release period.
