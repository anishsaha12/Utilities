'use strict';

let https = require('https');

// Replace the subscriptionKey string value with your valid subscription key.
let subscriptionKey = 'c10db6dc74b84ee5b878689332a9bc0e';


let host = 'api.cognitive.microsoft.com';
let path = '/bing/v7.0/search';

let term = 'Manoj Nuthakki';

let check_linkedin = function (webpage) {
    return (webpage.url.indexOf('linkedin.com') > 0)
}

let check_facebook = function (webpage) {
    return (webpage.url.indexOf('facebook.com') > 0)
}

let response_handler = function (response) {
    let body = '';
    response.on('data', function (d) {
        body += d;
    });
    response.on('end', function () {
        body = JSON.parse(body);
        let body_str = JSON.stringify(body, null, '  ');
        console.log('\nJSON Response:\n');
        // console.log(body_str);
        let webpage_list = body.webPages.value;
        let image_list = body.images.value;
        
        let linkedin_results = webpage_list.filter(check_linkedin);
        let facebook_results = webpage_list.filter(check_facebook);
        // console.log(linkedin_results)
    });
    response.on('error', function (e) {
        console.log('Error: ' + e.message);
    });
};

let bing_web_search = function (search) {
  console.log('Searching the Web for: ' + term);
  let request_params = {
        method : 'GET',
        hostname : host,
        path : path + '?q=' + encodeURIComponent(search),
        headers : {
            'Ocp-Apim-Subscription-Key' : subscriptionKey,
        }
    };

    let req = https.request(request_params, response_handler);
    req.end();
}

if (subscriptionKey.length === 32) {
    bing_web_search(term);
} else {
    console.log('Invalid Bing Search API subscription key!');
    console.log('Please paste yours into the source code.');
}