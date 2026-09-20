import React, { Component } from 'react';

export default class End extends Component {

    componentDidMount() {
        window.scrollTo(0, 0);
    }

    render() {
        return(
            <div>
                <div className="section">
                    <div className="content" style={{width: "900px"}}>
                        <h2>Thank you very much for your participation! The study is finished. Your data has been saved and you can close this window.</h2>
                    </div>
                </div>
            </div>
        );
    }

}