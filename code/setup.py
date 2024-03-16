''' This program is a form of manual exploratory data analysis; features are filtered and some territories (like Wallis and Futuna) are removed from the countries 
as to avoid redudancy (stats for Wallis and Futuna would be included in France's stats)'''

import json

# only UN recognized countries will be accepted (no territories)
recognized_countries = ["Afghanistan","Albania","Algeria","Andorra","Angola","Antigua and Barbuda","Argentina","Armenia","Australia","Austria","Azerbaijan","Bahamas, The","Bahrain","Bangladesh","Barbados","Belarus","Belgium","Belize","Benin","Bhutan","Bolivia","Bosnia and Herzegovina","Botswana","Brazil","Brunei","Bulgaria","Burkina Faso","Burundi","Cabo Verde","Cambodia","Cameroon","Canada","Central African Republic","Chad","Chile","China","Colombia","Comoros","Congo, Democratic Republic of the", "Congo, Republic of the", "Costa Rica","Cote d'Ivoire","Croatia","Cuba","Cyprus","Czechia","Denmark","Djibouti","Dominica","Dominican Republic","Ecuador","Egypt","El Salvador","Equatorial Guinea","Eritrea","Estonia","Eswatini","Ethiopia","Fiji","Finland","France","Gabon","Gambia, The","Georgia","Germany","Ghana","Greece","Grenada","Guatemala","Guinea","Guinea-Bissau","Guyana","Haiti","Honduras","Hungary","Iceland","India","Indonesia","Iran","Iraq","Ireland","Israel","Italy","Jamaica","Japan","Jordan","Kazakhstan","Kenya","Kiribati","Korea, North","Korea, South","Kuwait","Kyrgyzstan","Laos","Latvia","Lebanon","Lesotho","Liberia","Libya","Liechtenstein","Lithuania","Luxembourg","Madagascar","Malawi","Malaysia","Maldives","Mali","Malta","Marshall Islands","Mauritania","Mauritius","Mexico","Micronesia, Federated States of","Moldova","Monaco","Mongolia","Montenegro","Morocco","Mozambique","Myanmar","Namibia","Nauru","Nepal","Netherlands","New Zealand","Nicaragua","Niger","Nigeria","North Macedonia","Norway","Oman","Pakistan","Palau","Palestine","Panama","Papua New Guinea","Paraguay","Peru","Philippines","Poland","Portugal","Qatar","Romania","Russia","Rwanda","Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines","Samoa","San Marino","Sao Tome and Principe","Saudi Arabia","Senegal","Serbia","Seychelles","Sierra Leone","Singapore","Slovakia","Slovenia","Solomon Islands","Somalia","South Africa","South Sudan","Spain","Sri Lanka","Sudan","Suriname","Sweden","Switzerland", "Syria", "Taiwan","Tajikistan","Tanzania","Thailand","Timor-Leste","Togo","Tonga","Trinidad and Tobago","Tunisia","Turkey (Turkiye)","Turkmenistan","Tuvalu","Uganda","Ukraine","United Arab Emirates","United Kingdom","United States","Uruguay","Uzbekistan","Vanuatu","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe"] # "Vatican City"

# function to get desired features and countries and write to 'dataset.json'
def clean(data, list_features):

    features = {}

    # for all countries and their attributes in the original dataset 'countries.json', loop through and select only desired features
    for entry, attributes in data.items():

        # if the country is UN recognized
        if entry in recognized_countries:
            desired_features = {}

            # now, go and add the desired features
            for k in list_features: 

                # format all attributes into numerical data if possible, else impute with placeholder
                try:
                    substring = attributes.get(k, None).replace(",", "")
                    space_idx = substring.find(' ')
                    percent_idx = substring.find('%')
                    end_idx = len(substring)

                    if space_idx != -1 and percent_idx != -1:
                        end_idx = min(space_idx, percent_idx)
                    elif space_idx != -1:
                        end_idx = space_idx
                    elif percent_idx != -1:
                        end_idx = percent_idx

                    # add to dictionary to write to dataset later
                    try:
                        desired_features[f"{k}"] = float(substring[0:end_idx])
                    except ValueError:
                        desired_features[f"{k}"] = 0
                        print(f"{entry} has no value for {k}. 0 is set as placeholder value.")

                # catch errors and impute
                except AttributeError or TypeError:
                    desired_features[f"{k}"] = 0
                    print(f"{entry} has no key named {k}. Key is created and 0 is set as placeholder value.")

            features[entry] = desired_features

    return features

def main():
    # read in list_features
    with open('list_features.txt', 'r') as file:
        list_features = [line.strip() for line in file]

    list_features.append("People and Society: Population growth rate")

    # read in original dataset
    with open('countries.json', 'r') as file:
        data = json.load(file)
    
    # call clean and filter out unnecessary features
    cleaned_dataset = clean(data, list_features)

    # write trimmed dataset to 'dataset.json'
    with open('dataset.json', 'w') as file:
        json.dump(cleaned_dataset, file, indent=2)

if __name__ == "__main__":
    main()
