# DDI-subnet-clenup
code calling men and mice API and adding additional features to get a full picture for cleaning the network from unused subnets.
attached the code itself in .py code and an .html file to view example results

**code flow:**
1. imports, connections to sql dbs (for data saving), auxilary functions
2. API call - get all data from IPAM tool concerning used and reserved subnets
3. collecting additional relevant data and combining it to one table
4. deciding on 4 types of usage, will be regarded from now on as _labels_: 
  - free: subnet is in the allowed /16 subnet area but was not reserved
  - never discovered: was given for a specific usage but never used
  - last seen old: was given nand used but not for a long time (default is a year)
  - in use: was used lately (default in the last year)
  each subnet in the dataset will get one of the above labels
5. geting rid of the subnets that are larger than /16 and smaller than /29 in the db. avoiding clutter and non representativeness (too large/small subnets)
6. assigning a /16 predecessor to every subnet and removing irrelevant /16 segments (3.3.3.3 and other "garbag" subnets)
7. finding immediate parent of subnet from within the dataset
8. finding free subnets that are "leaf nodes" of existing reserved subnets
9. assigning "is_leaf" flag for every subnet: is this node is a leaf in the subnet tree?
10. creating immediate children dictionary and full ancestor/predecessor dictionary for easyt access
11. assigning site_code, environment and campus_code for every subnet. if there are multiple, looking at the subnet's children, there will be a list of the relevant ones.
12. free subnets will get site_code, environment and campus_code of its immediate parent
13. assigning whether a subnet is at the top of their site_code, environment and campus_code respectively (non dependant empirically): this feature will lead to easier lookup in future tasks
14. _db upload of the edited table - full_ipam_fragmentation_
15. for every /16 creating a summary table pointing the allocation of this subnet's leaf children. This will diverse according to the labels and environment. each site code\campus code will have it's own data for further filtering conveniency
16. _db upload of the edited table - sixteens_summary_
17. creating heritage table consisting every ancestor/predecessor relationship in the data, including but not exclusive to immediate relationships
18. marging/joining the relevant tables together to create a more insightful data table
19. creating a similar summary table, now using all of the data that is at the top of their site_code or environment or campus_code
20. _db upload of the edited table - top_summary
