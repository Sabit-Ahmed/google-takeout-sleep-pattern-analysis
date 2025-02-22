import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

const SleepAnalysis = () => {
  const data = [
    { hour: '00:00', inactivePercentage: 85.7, avgDuration: 0 },
    { hour: '01:00', inactivePercentage: 57.1, avgDuration: 8.8 },
    { hour: '02:00', inactivePercentage: 100, avgDuration: 0 },
    { hour: '03:00', inactivePercentage: 100, avgDuration: 0 },
    { hour: '04:00', inactivePercentage: 100, avgDuration: 0 },
    { hour: '05:00', inactivePercentage: 100, avgDuration: 0 },
    { hour: '06:00', inactivePercentage: 100, avgDuration: 0 },
    { hour: '07:00', inactivePercentage: 100, avgDuration: 0 },
    { hour: '08:00', inactivePercentage: 100, avgDuration: 0 },
    { hour: '09:00', inactivePercentage: 85.7, avgDuration: 0 },
    { hour: '10:00', inactivePercentage: 42.9, avgDuration: 0 },
    { hour: '11:00', inactivePercentage: 14.3, avgDuration: 0 },
    { hour: '12:00', inactivePercentage: 28.6, avgDuration: 3.3 },
    { hour: '13:00', inactivePercentage: 42.9, avgDuration: 2.6 },
    { hour: '14:00', inactivePercentage: 71.4, avgDuration: 0 },
    { hour: '15:00', inactivePercentage: 42.9, avgDuration: 2.9 },
    { hour: '16:00', inactivePercentage: 28.6, avgDuration: 3.7 },
    { hour: '17:00', inactivePercentage: 57.1, avgDuration: 2.9 },
    { hour: '18:00', inactivePercentage: 85.7, avgDuration: 0 },
    { hour: '19:00', inactivePercentage: 57.1, avgDuration: 0 },
    { hour: '20:00', inactivePercentage: 14.3, avgDuration: 0 },
    { hour: '21:00', inactivePercentage: 28.6, avgDuration: 4.0 },
    { hour: '22:00', inactivePercentage: 28.6, avgDuration: 12.4 },
    { hour: '23:00', inactivePercentage: 42.9, avgDuration: 10.5 }
  ];

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Sleep Pattern Analysis (Jan 19-25, 2025)</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart 
              data={data}
              margin={{ top: 20, right: 60, left: 60, bottom: 20 }}
            >
              <XAxis 
                dataKey="hour" 
                label={{ value: 'Hour of Day', position: 'bottom', offset: 0 }}
              />
              <YAxis 
                yAxisId="left"
                label={{ value: '% Days Inactive', angle: -90, position: 'insideLeft', offset: -40 }}
              />
              <YAxis 
                yAxisId="right"
                orientation="right"
                label={{ value: 'Avg Hours When Inactive', angle: 90, position: 'insideRight', offset: -40 }}
              />
              <Tooltip />
              <Legend wrapperStyle={{ position: 'relative', marginTop: '10px' }}/>
              <Bar yAxisId="left" dataKey="inactivePercentage" fill="#8884d8" name="% of Days Inactive" />
              <Bar yAxisId="right" dataKey="avgDuration" fill="#82ca9d" name="Avg Duration When Inactive (hours)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-8 space-y-2">
          <p className="text-sm text-gray-600">
            This chart shows two key metrics:
            1. Blue bars: The percentage of days where each hour showed no activity (likely sleeping)
            2. Green bars: When inactivity started at this hour, how long it lasted on average
          </p>
          <p className="text-sm text-gray-600">
            Key findings:
            - Most consistent sleep hours: 2-8 AM (100% inactive)
            - Longest sleep periods start between 10 PM-1 AM (8-12 hours)
            - Afternoon inactivity periods (2-4 hours) might indicate breaks/naps
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default SleepAnalysis;