const fs = require('fs');
const path = require('path');

function main() {
    const dataPath = process.argv[2];
    const trainPath = process.argv[3];
    const testPath = process.argv[4];

    fs.mkdirSync(testPath, {recursive: true});
    fs.mkdirSync(trainPath, {recursive: true});

    const countries = fs.readdirSync(dataPath);
    countries.forEach(country => {
        const slug = country.trim().toLowerCase().replace(/\s/g, '_');

        const images = fs.readdirSync(path.join(dataPath, country));
        let testCount = 0;
        let trainCount = 0;

        for (let i = 0; i < images.length; i++) {
            const imgName = images[i];
            const imgPath = path.join(dataPath, country, imgName);
            const extension = path.extname(imgName);
            const isTest = Math.random() > 0.8;

            let newPath = '';
            if (isTest) {
                newPath = path.join(testPath, `${slug}_${testCount.toString().padStart(5, '0')}${extension}`);
                testCount++;
            } else {
                newPath = path.join(trainPath, `${slug}_${trainCount.toString().padStart(5, '0')}${extension}`);
                trainCount++;
            }

            console.log(`Renaming ${imgPath} to ${newPath}`);
            // fs.renameSync(imgPath, newPath);
        }

        console.log(`${country}: ${trainCount} ${testCount}`);
    });
}

main();
