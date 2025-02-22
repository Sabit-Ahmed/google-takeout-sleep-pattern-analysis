import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import Papa from 'papaparse';

const getColorForValue = (value) => {
    if (value === 0) return '#f3f4f6';  
    if (value < 0.05) return '#fee2e2';  
    if (value < 0.10) return '#fecaca';  
    if (value < 0.15) return '#ef4444';  
    if (value < 0.20) return '#dc2626';  
    if (value < 0.30) return '#b91c1c';  
    return '#7f1d1d';  
};

const TopicHeatmap = () => {
    const [data, setData] = useState(null);
    const cellWidth = 35; // Reduced cell width to fit screen better

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await window.fs.readFile('categorized_output.csv', { encoding: 'utf8' });
                const parsedData = Papa.parse(response, {
                    header: true,
                    skipEmptyLines: true
                });

                const activities = parsedData.data.map(row => ({
                    ...row,
                    timestamp: new Date(row.timestamp),
                    hour: new Date(row.timestamp).getHours()
                }));

                const topicData = {};
                activities.forEach(activity => {
                    if (!topicData[activity.topic]) {
                        topicData[activity.topic] = {
                            total: 0,
                            hours: Array(24).fill(0)
                        };
                    }
                    topicData[activity.topic].total++;
                    topicData[activity.topic].hours[activity.hour]++;
                });

                const significantTopics = Object.entries(topicData)
                    .filter(([, data]) => data.total >= 20)
                    .sort(([,a], [,b]) => b.total - a.total)
                    .reduce((acc, [topic, data]) => {
                        acc[topic] = data;
                        return acc;
                    }, {});

                setData(significantTopics);
            } catch (error) {
                console.error('Error loading data:', error);
            }
        };

        fetchData();
    }, []);

    if (!data) return <div>Loading...</div>;

    return (
        <div className="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>Topic Activity Heatmap (Topics with 20+ occurrences)</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="w-full overflow-x-auto">
                        <div style={{ minWidth: '1100px', maxWidth: '1300px', margin: '0 auto' }}>
                            <div className="relative">
                                {/* Topic rows with heatmap cells */}
                                {Object.entries(data).map(([topic, { hours, total }]) => (
                                    <div key={topic} className="mb-4">
                                        <div className="flex items-center">
                                            <div className="w-60 text-sm font-medium pr-4">
                                                {topic}
                                                <span className="text-gray-500 text-xs ml-1">({total})</span>
                                            </div>
                                            <div className="flex">
                                                {hours.map((count, hour) => {
                                                    const proportion = count / total;
                                                    return (
                                                        <div
                                                            key={hour}
                                                            style={{
                                                                width: `${cellWidth}px`,
                                                                backgroundColor: getColorForValue(proportion),
                                                            }}
                                                            className="h-8 border-r border-white relative group"
                                                        >
                                                            <div className="absolute hidden group-hover:block bg-black text-white text-xs p-1 rounded -top-8 left-1/2 transform -translate-x-1/2 whitespace-nowrap z-10">
                                                                {hour}:00 - {count} activities ({(proportion * 100).toFixed(1)}%)
                                                            </div>
                                                            <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                                                                {proportion >= 0.1 ? `${(proportion * 100).toFixed(0)}%` : ''}
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    </div>
                                ))}

                                {/* X-axis labels */}
                                <div className="flex mt-8 pl-60"> {/* Increased top margin */}
                                    {Array.from({ length: 24 }, (_, i) => (
                                        <div
                                            key={i}
                                            style={{
                                                width: `${cellWidth}px`,
                                                marginLeft: '0px',
                                                marginTop: '20px' // Added more spacing
                                            }}
                                            className="text-xs transform -rotate-45 origin-top-left flex justify-center"
                                        >
                                            {i.toString().padStart(2, '0')}:00
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Legend */}
                            <div className="mt-20 flex items-center justify-center space-x-4">
                                <div className="text-sm">Activity Level:</div>
                                {[0, 0.05, 0.1, 0.15, 0.2, 0.3].map((value, i) => (
                                    <div key={i} className="flex items-center">
                                        <div 
                                            className="w-6 h-6 mr-1" 
                                            style={{ backgroundColor: getColorForValue(value) }}
                                        />
                                        <span className="text-xs">{`${(value * 100)}%`}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Summary Statistics */}
            <Card>
                <CardHeader>
                    <CardTitle>Peak Activity Hours by Topic</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {Object.entries(data).map(([topic, { hours, total }]) => {
                            const peakHour = hours.indexOf(Math.max(...hours));
                            const peakPercentage = ((hours[peakHour] / total) * 100).toFixed(1);
                            
                            return (
                                <div key={topic} className="p-4 border rounded-lg">
                                    <h3 className="font-medium mb-2">{topic}</h3>
                                    <div className="text-sm">
                                        <div>Peak hour: {peakHour}:00</div>
                                        <div>Peak activity: {peakPercentage}% of daily total</div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};

export default TopicHeatmap;