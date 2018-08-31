const program = require('commander');
const rp = require('request-promise');
var fs = require('fs');

const checkResourceFormat = (format) => {
	//console.log(format.toLowerCase());

	if (format.toLowerCase() === "json"){
		return true;
	}
	if (format.toLowerCase() === "csv"){
		return true;
	}	


	return false;
}

const processDatasetResponses = (datasetResponses) => {
	console.log("processDatasetResponses datasetResponses.length " + datasetResponses.length);

	var responseObject = {}
	for(var i = 0; i < datasetResponses.length; i++) {
		var datasetResponse = datasetResponses[i];
		for (var j = 0; j < datasetResponse.body.result.results.length; j++){
			

			if (datasetResponse.body.result.results[j].name !== datasetResponse.datasetName) {
				// console.log(datasetResponse.body.result.results[j].name + " !== " + datasetResponse.datasetName);
				continue;
			}

			// console.log(datasetResponse.body.result.results[j].name + " === " + datasetResponse.datasetName);
			// console.log(datasetResponse.body.result.results[j].name + " " + datasetResponse.body.result.results[j].resources.length);

			var resourceObjList = []
			for (var k = 0; k < datasetResponse.body.result.results[j].resources.length; k++){
				var resource = datasetResponse.body.result.results[j].resources[k];

				if (!checkResourceFormat(resource.format)){
					continue;
				}

				var resourceObj = {
					name: resource.name.toLowerCase(),
					format: resource.format.toLowerCase(),
					url: resource.url,
				}

				//console.log(JSON.stringify(resourceObj));
				resourceObjList.push(resourceObj);
			}

			var tagObjList = []
			for (var k = 0; k < datasetResponse.body.result.results[j].tags.length; k++){
				var tag = datasetResponse.body.result.results[j].tags[k];

				var tagObj = {
					name: tag.name.toLowerCase(),
					display_name: tag.display_name.toLowerCase(),
				}
				tagObjList.push(tagObj);						
			}

			var groupObjList = []
			for (var k = 0; k < datasetResponse.body.result.results[j].groups.length; k++){
				var group = datasetResponse.body.result.results[j].groups[k];

				var groupObj = {
					name: group.name.toLowerCase(),
					display_name: group.display_name.toLowerCase(),
					description: group.description.toLowerCase(),
				}
				groupObjList.push(groupObj);							
			}

			var datasetObj = {
				title: datasetResponse.body.result.results[j].title.toLowerCase(),
				notes: datasetResponse.body.result.results[j].notes.toLowerCase(),
				count: datasetResponse.body.result.count,

				resources: resourceObjList,
				tags: tagObjList,
				groups: groupObjList,
			}

			responseObject[datasetResponse.datasetName] = datasetObj;
			//console.log(JSON.stringify(responseObject[datasetResponse.datasetName]));
		}
	}	

	//console.log(JSON.stringify(responseObject));
	return responseObject;
}

const requestPromise = (req) => {
  return rp({
    uri: req,
    json: true,
    resolveWithFullResponse: true
  });
}

const requestDataset = (req) => {
	//console.log("http://data.surrey.ca/api/3/action/package_search?q=" + req);
	return rp({
	    uri: "http://data.surrey.ca/api/3/action/package_search?q=" + req,
	    json: true,
	    resolveWithFullResponse: true
  	}).then( (response) => {
  		//console.log("http://data.surrey.ca/api/3/action/package_search?q=" + req);
  		response.datasetName = req;
  		console.log(response.datasetName);
  		return response;
  	}).catch(function(err){
			console.log("requestDataset error " + err); 
	});
}

const requestDataCatalogue = function() {
  requestPromise("http://data.surrey.ca/api/3/action/package_list")
  .then( (response) => {
    	console.log("response.length: " + response.body.result.length);
		// console.log("response.count: " + response.body.result.count);    	

		var promises = Promise.all(response.body.result.map(requestDataset));

		promises.then( (datasetResponses) => {
			console.log("datasetResponses.length: " + datasetResponses.length);
			return processDatasetResponses(datasetResponses);
		})
		.then( (responseObject) => {
			var responseStr = JSON.stringify(responseObject);
			//console.log("response: " + responseStr);
			fs.writeFile("./downloadResourceURL.json", responseStr, function(err) {
			    if(err) {
			        return console.log(err);
			    }
			});  			
			return responseStr;			
		}).catch(function(err){
			console.log("error " + err); 
		});

  });	
}

const surreyGetData = function() {
	return new Promise((resolve, reject) => {
		requestDataCatalogue((err, result) => {
		      if (err) {
		        return reject(err)
		      }
		      return resolve(result)
		});
	});
}

program
  .version('1');

// node index.js downloadCatalogue
program
  .command('downloadCatalogue')
  .description('Project: Schema Integration for Heterogenous Data Sources')
  .action(function(){
    //console.log('calling surreyGetData()');

    surreyGetData()
    
  })

program.parse(process.argv);

