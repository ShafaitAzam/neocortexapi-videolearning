using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Azure.Storage.Queues;
using Azure.Storage.Blobs;
using Azure.Storage.Queues.Models;
using Microsoft.WindowsAzure.Storage;
using Microsoft.WindowsAzure.Storage.Auth;
using Microsoft.WindowsAzure.Storage.Table;

namespace HTMVideoLearning
{
    internal class AzureConnector
    {
        /// <summary>
        /// This is the string to connect to the storage account
        /// </summary>
        private string queConnectionString { get; set; }
        private string StorageAccountKey { get; set; }
        private string StorageAccountName { get; set; }
        public CloudTableClient tableClient { get; set; }

        /// <summary>
        /// Constructor to create the azure account connector
        /// </summary>
        /// <param name="queConnectionString">The string to connect to the que storage</param>
        public AzureConnector(string StorageAccountKey, string StorageAccountName, string queConnectionString)
        {
            this.queConnectionString = queConnectionString; 
            this.StorageAccountKey = StorageAccountKey;
            this.StorageAccountName = StorageAccountName;

            var storageAccount = new CloudStorageAccount(new StorageCredentials(StorageAccountName, StorageAccountKey), true);
            this.tableClient = storageAccount.CreateCloudTableClient();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="queueName"></param>
        /// <returns></returns>
        public async Task CreateQueue(string queueName)
        {
            QueueClient queue = new QueueClient(queConnectionString, queueName);
            await queue.CreateIfNotExistsAsync();
        }

        public static async Task CreateTable(string TableName)
        {
            var table = tableClient.GetTableReference(TableName);
            table.CreateIfNotExistsAsync();
            Console.WriteLine($"Table named {TableName} is created.");
        }

        /// <summary>
        /// This function will upload the files to blobstorage
        /// </summary>
        /// <param name="blobStorageConnectionString">The string to connect to the blob storage</param>
        /// <param name="blobStorageContainerName">Name of the blob storage container</param>
        /// <param name="trainingVideoPath">Video path in the local computer</param>
        /// <returns></returns>
        public async Task BlobStorageUpload(string blobStorageConnectionString, string blobStorageContainerName, string trainingVideoPath)
        {
            var container = new BlobContainerClient(blobStorageConnectionString, blobStorageContainerName);
            await container.CreateIfNotExistsAsync();

            string fileName = Path.GetFileName(trainingVideoPath);
            var blobClient = container.GetBlobClient(fileName);

            if (blobClient.Exists() == false)
            {
                var stream = File.OpenRead(trainingVideoPath);
                await blobClient.UploadAsync(stream);
                Console.WriteLine($" {fileName} Upload completed.");
            }

            else 
            {
                Console.WriteLine($"{fileName} Already exists");
            }            
        }
    }
}
