import xmltodict
import csv
import pandas as pd
import sys

#open the file
pn = '90247161'
fileptr = open(f"temp_store_xml/{pn}.xml","r")

#read xml content from the file
xml_content= fileptr.read()
#print("XML content is:")
#print(xml_content)

my_ordered_dict=xmltodict.parse(xml_content)
print("Start Process:")
#print(my_ordered_dict)

#store all dictionaries

# Method 1

df1 = pd.DataFrame(my_ordered_dict['EXPORT']['FKBearbeitungsGruppe'],list(range(len(my_ordered_dict['EXPORT']['FKBearbeitungsGruppe']))))
df2 = pd.DataFrame(my_ordered_dict['EXPORT']['FKBaum'],list(range(len(my_ordered_dict['EXPORT']['FKBaum']))))
df3 = pd.DataFrame(my_ordered_dict['EXPORT']['FKBearbeitungsEinheit'],list(range(len(my_ordered_dict['EXPORT']['FKBearbeitungsEinheit']))))
df4 = pd.DataFrame(my_ordered_dict['EXPORT']['FKWZZuordKonfig'],list(range(len(my_ordered_dict['EXPORT']['FKWZZuordKonfig']))))
df5 = pd.DataFrame(my_ordered_dict['EXPORT']['FKStiftDornWZ'],list(range(len(my_ordered_dict['EXPORT']['FKStiftDornWZ']))))
df1.to_csv(f'xml_processed/FKBearbeitungsGruppe_{pn}_c.csv', index=False, header=True)
df2.to_csv(f'xml_processed/FKBaum_{pn}_c.csv', index=False, header=True)
df3.to_csv(f'xml_processed/FKBearbeitungsEinheit_{pn}_c.csv', index=False, header=True)
df4.to_csv(f'xml_processed/FKWZZuordKonfig_{pn}_c.csv', index=False, header=True)
df5.to_csv(f'xml_processed/FKStiftDornWZ_{pn}_c.csv', index=False, header=True)


df8 = pd.DataFrame(my_ordered_dict['EXPORT']['FKRille'],list(range(len(my_ordered_dict['EXPORT']['FKRille']))))
df16 = pd.DataFrame(my_ordered_dict['EXPORT']['FKTeil'],list(range(len(my_ordered_dict['EXPORT']['FKTeil']))))
df20 = pd.DataFrame(my_ordered_dict['EXPORT']['FKTeilGlobDaten'],list(range(len(my_ordered_dict['EXPORT']['FKTeilGlobDaten']))))
df21 = pd.DataFrame(my_ordered_dict['EXPORT']['FKEinstellungen'],list(range(len(my_ordered_dict['EXPORT']['FKEinstellungen']))))
df22 = pd.DataFrame(my_ordered_dict['EXPORT']['FKEinstellWert'],list(range(len(my_ordered_dict['EXPORT']['FKEinstellWert']))))
df8.to_csv(f'xml_processed/FKRille_{pn}_c.csv', index=False, header=True)
df16.to_csv(f'xml_processed/FKTeil_{pn}_c.csv', index=False, header=True)
df20.to_csv(f'xml_processed/FKTeilGlobDaten_{pn}_c.csv', index=False, header=True)
df21.to_csv(f'xml_processed/FKEinstellungen_{pn}_c.csv', index=False, header=True)
df22.to_csv(f'xml_processed/FKEinstellWert_{pn}_c.csv', index=False, header=True)



for i in range(0,len(my_ordered_dict['EXPORT']['FKWZKatalog'])):
	df7 = pd.DataFrame(my_ordered_dict['EXPORT']['FKWZKatalog'][i],list(range(len(my_ordered_dict['EXPORT']['FKWZKatalog'][i]))))
	if i==0:
		df7.to_csv(f'xml_processed/FKWZKatalog_{pn}_c.csv', index=False, header=True)
	else:
		df7.to_csv(f'xml_processed/FKWZKatalog_{pn}_c.csv', mode='a',index=False, header=False)


for i in range(0,len(my_ordered_dict['EXPORT']['FKRillenElement'])):	
	df9 = pd.DataFrame(my_ordered_dict['EXPORT']['FKRillenElement'][i],list(range(len(my_ordered_dict['EXPORT']['FKRillenElement'][i]))))
	if i==0:
		df9.to_csv(f'xml_processed/FKRillenElement_{pn}_c.csv', index=False, header=True)
	else:
		df9.to_csv(f'xml_processed/FKRillenElement_{pn}_c.csv', mode='a',index=False, header=False)

for i in range(0,len(my_ordered_dict['EXPORT']['FKMaterial'])):
	df10 = pd.DataFrame(my_ordered_dict['EXPORT']['FKMaterial'][i],list(range(len(my_ordered_dict['EXPORT']['FKMaterial'][i]))))
	if i==0:
		df10.to_csv(f'xml_processed/FKMaterial_{pn}_c.csv', index=False, header=True)
	else:
		df10.to_csv(f'xml_processed/FKMaterial_{pn}_c.csv', mode='a',index=False, header=False)

for i in range(0,len(my_ordered_dict['EXPORT']['FKMatKatalog'])):
	df11 = pd.DataFrame(my_ordered_dict['EXPORT']['FKMatKatalog'][i],list(range(len(my_ordered_dict['EXPORT']['FKMatKatalog'][i]))))
	if i==0:
		df11.to_csv(f'xml_processed/FKMatKatalog_{pn}_c.csv', index=False, header=True)
	else:
		df11.to_csv(f'xml_processed/FKMatKatalog_{pn}_c.csv', mode='a',index=False, header=False)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKKennlinie'])):
	df12 = pd.DataFrame(my_ordered_dict['EXPORT']['FKKennlinie'][i],list(range(len(my_ordered_dict['EXPORT']['FKKennlinie'][i]))))
	if i==0:
		df12.to_csv(f'xml_processed/FKKennlinie_{pn}_c.csv', index=False, header=True)
	else:
		df12.to_csv(f'xml_processed/FKKennlinie_{pn}_c.csv', mode='a',index=False, header=False)


	
for i in range(0,len(my_ordered_dict['EXPORT']['FKKLKatalog'])):
	df13 = pd.DataFrame(my_ordered_dict['EXPORT']['FKKLKatalog'][i],list(range(len(my_ordered_dict['EXPORT']['FKKLKatalog'][i]))))
	if i==0:
		df13.to_csv(f'xml_processed/FKKLKatalog_{pn}_c.csv', index=False, header=True)
	else:
		df13.to_csv(f'xml_processed/FKKLKatalog_{pn}_c.csv', mode='a',index=False, header=False)


	
for i in range(0,len(my_ordered_dict['EXPORT']['FKWZAblaufdaten'])):
	df14 = pd.DataFrame(my_ordered_dict['EXPORT']['FKWZAblaufdaten'][i],list(range(len(my_ordered_dict['EXPORT']['FKWZAblaufdaten'][i]))))
	if i==0:
		df14.to_csv(f'xml_processed/FKWZAblaufdaten_{pn}_c.csv', index=False, header=True)
	else:
		df14.to_csv(f'xml_processed/FKWZAblaufdaten_{pn}_c.csv', mode='a',index=False, header=False)


for i in range(0,len(my_ordered_dict['EXPORT']['FKWZAblaufdatenKatalog'])):
	df15 = pd.DataFrame(my_ordered_dict['EXPORT']['FKWZAblaufdatenKatalog'][i],list(range(len(my_ordered_dict['EXPORT']['FKWZAblaufdatenKatalog'][i]))))
	if i==0:
		df15.to_csv(f'xml_processed/FKWZAblaufdatenKatalog_{pn}_c.csv', index=False, header=True)
	else:
		df15.to_csv(f'xml_processed/FKWZAblaufdatenKatalog_{pn}_c.csv', mode='a',index=False, header=False)

	
for i in range(0,len(my_ordered_dict['EXPORT']['FKTeileKatalog'])):
	df17 = pd.DataFrame(my_ordered_dict['EXPORT']['FKTeileKatalog'][i],list(range(len(my_ordered_dict['EXPORT']['FKTeileKatalog'][i]))))
	if i==0:
		df17.to_csv(f'xml_processed/FKTeileKatalog_{pn}_c.csv', index=False, header=True)
	else:
		df17.to_csv(f'xml_processed/FKTeileKatalog_{pn}_c.csv', mode='a',index=False, header=False)


for i in range(0,len(my_ordered_dict['EXPORT']['FKGeoElement'])):
	df18 = pd.DataFrame(my_ordered_dict['EXPORT']['FKGeoElement'][i],list(range(len(my_ordered_dict['EXPORT']['FKGeoElement'][i]))))
	if i==0:
		df18.to_csv(f'xml_processed/FKGeoElement_{pn}_c.csv', index=False, header=True)
	else:
		df18.to_csv(f'xml_processed/FKGeoElement_{pn}_c.csv', mode='a',index=False, header=False)

for i in range(0,len(my_ordered_dict['EXPORT']['FKMakroZeile'])):
	df19 = pd.DataFrame(my_ordered_dict['EXPORT']['FKMakroZeile'][i],list(range(len(my_ordered_dict['EXPORT']['FKMakroZeile'][i]))))
	if i==0:
		df19.to_csv(f'xml_processed/FKMakroZeile_{pn}_c.csv', index=False, header=True)
	else:
		df19.to_csv(f'xml_processed/FKMakroZeile_{pn}_c.csv', mode='a',index=False, header=False)
	

for i in range(0,len(my_ordered_dict['EXPORT']['FKGlobWert'])):
	df23 = pd.DataFrame(my_ordered_dict['EXPORT']['FKGlobWert'][i],list(range(len(my_ordered_dict['EXPORT']['FKGlobWert'][i]))))
	if i==0:
		df23.to_csv(f'xml_processed/FKGlobWert_{pn}_c.csv', index=False, header=True)
	else:
		df23.to_csv(f'xml_processed/FKGlobWert_{pn}_c.csv', mode='a',index=False, header=False)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKFreiesElement'])):
	df24 = pd.DataFrame(my_ordered_dict['EXPORT']['FKFreiesElement'][i],list(range(len(my_ordered_dict['EXPORT']['FKFreiesElement'][i]))))
	if i==0:
		df24.to_csv(f'xml_processed/FKFreiesElement_{pn}_c.csv', index=False, header=True)
	else:
		df24.to_csv(f'xml_processed/FKFreiesElement_{pn}_c.csv', mode='a',index=False, header=False)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKKoordWert'])):
	df25 = pd.DataFrame(my_ordered_dict['EXPORT']['FKKoordWert'][i],list(range(len(my_ordered_dict['EXPORT']['FKKoordWert'][i]))))
	if i==0:
		df25.to_csv(f'xml_processed/FKKoordWert_{pn}_c.csv', index=False, header=True)
	else:
		df25.to_csv(f'xml_processed/FKKoordWert_{pn}_c.csv', mode='a',index=False, header=False)


print("Finished Processing")















