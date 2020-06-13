



#Use the below for the real data
bins= [0,0.35,0.7,1]
plt.hist(df.values, bins=bins, edgecolor="k")
plt.xticks(bins)

plt.show()